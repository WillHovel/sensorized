#!/usr/bin/env python3
"""
Gesture recognizer control panel.

One Tkinter app covering the whole workflow:
  - Gestures tab   : edit the gesture list in gesture_config.h
  - Collect tab    : stream labeled sensor data from the ESP32 to a CSV
  - Train tab      : run train_model.py, see the tuning steps, save/reload
                      named trainings, deploy one to the Arduino library
  - Run tab        : run the deployed model live, adjust polling rate on
                      the fly, optionally remap gesture names to custom
                      meanings (e.g. "fist" -> "Resting"), and export the
                      session log as a CSV ready to open in a spreadsheet

Dependencies beyond the standard library: pyserial (always), and bleak if
you want the BLE connection mode (`pip install -r requirements.txt` gets
both). USB mode works without bleak installed - BLE mode just won't.

Connect over USB or BLE (pick with the radio buttons at the top) to
whichever sketch is currently flashed on the board - this one app talks to
both 01_data_collection (for the Collect tab) and 03_inference (for the Run
tab), since it tells them apart by the shape of each line. Note:
01_data_collection doesn't have BLE firmware - only 03_inference does, so
use USB for data collection and either for running the trained model.
"""
import asyncio
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, simpledialog

import serial
import serial.tools.list_ports

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False

# Must match the UUIDs in 03_inference.ino
BLE_SERVICE_UUID = "6f23cf20-1efb-4a2e-9d35-9d1b6a8d8d01"
BLE_DATA_CHAR_UUID = "6f23cf20-1efb-4a2e-9d35-9d1b6a8d8d02"
BLE_CMD_CHAR_UUID = "6f23cf20-1efb-4a2e-9d35-9d1b6a8d8d03"
BLE_DEVICE_NAME = "GestureRecognizer"

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE.parent / "02_train_model" / "train_model.py"
SAVED_TRAININGS_DIR = HERE / "saved_trainings"
SETTINGS_PATH = HERE / "gui_settings.json"
CURRENT_MODEL_PATH = HERE / "model_weights.h"  # overwritten by each Train run

SAVED_TRAININGS_DIR.mkdir(exist_ok=True)

DEFAULT_SETTINGS = {
    "gesture_config_path": "",
    "library_model_path": "",
    "last_port": "",
    "link_mode": "USB",
    "ble_name": BLE_DEVICE_NAME,
    "dataset_path": str(HERE / "dataset.csv"),
    "mappings": {},
}


def load_settings():
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text())
            return {**DEFAULT_SETTINGS, **data}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings):
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# gesture_config.h parsing / rewriting
# ---------------------------------------------------------------------------
GESTURE_BLOCK_RE = re.compile(
    r"(static const char\* const GESTURE_LABELS\[\]\s*=\s*\{)(.*?)(\};)", re.DOTALL
)


def read_gesture_labels(path):
    text = Path(path).read_text()
    m = GESTURE_BLOCK_RE.search(text)
    if not m:
        raise ValueError("Couldn't find GESTURE_LABELS[] in that file - is it the right gesture_config.h?")
    return re.findall(r'"([^"]*)"', m.group(2))


def write_gesture_labels(path, labels):
    text = Path(path).read_text()
    m = GESTURE_BLOCK_RE.search(text)
    if not m:
        raise ValueError("Couldn't find GESTURE_LABELS[] in that file - is it the right gesture_config.h?")
    body = "\n" + "".join(f'    "{name}",\n' for name in labels)
    new_text = text[: m.start()] + m.group(1) + body + m.group(3) + text[m.end():]
    Path(path).write_text(new_text)


# ---------------------------------------------------------------------------
# Serial link - background reader thread feeding a thread-safe queue
# ---------------------------------------------------------------------------
class SerialLink:
    def __init__(self):
        self.ser = None
        self.q = queue.Queue()
        self.thread = None
        self.stop_flag = threading.Event()

    @property
    def connected(self):
        return self.ser is not None and self.ser.is_open

    def connect(self, port, baud=115200):
        self.disconnect()
        self.ser = serial.Serial(port, baud, timeout=0.2)
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stop_flag.is_set():
            try:
                raw = self.ser.readline()
            except Exception:
                break
            if raw:
                line = raw.decode(errors="ignore").strip()
                if line:
                    self.q.put(line)

    def send(self, text):
        if self.connected:
            self.ser.write((text + "\n").encode())

    def disconnect(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=1)
        self.thread = None
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None


class BLELink:
    """Same interface as SerialLink (q / connected / connect / send /
    disconnect) so the rest of the app doesn't need to know which one it's
    talking to. Runs its own asyncio loop in a background thread since
    bleak is async-only."""

    def __init__(self):
        self.q = queue.Queue()
        self.loop = None
        self.thread = None
        self.client = None
        self._connected = False
        self._stop = threading.Event()

    @property
    def connected(self):
        return self._connected

    def connect(self, target):
        """target: a device name to scan for, or a BLE address if you already know it."""
        self.disconnect()
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, args=(target,), daemon=True)
        self.thread.start()

    def _run(self, target):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._connect_and_listen(target))
        except Exception as e:
            self.q.put(f"# BLE error: {e}")
        self._connected = False

    async def _connect_and_listen(self, target):
        device = target
        if ":" not in target:  # treat as a name to scan for, not an address
            self.q.put(f"# scanning for '{target}'...")
            found = await BleakScanner.find_device_by_name(target, timeout=10.0)
            if not found:
                self.q.put(f"# BLE device '{target}' not found")
                return
            device = found

        async with BleakClient(device) as client:
            self.client = client
            self._connected = True
            self.q.put("# BLE connected")

            def handler(_sender, data):
                try:
                    self.q.put(data.decode(errors="ignore").strip())
                except Exception:
                    pass

            await client.start_notify(BLE_DATA_CHAR_UUID, handler)
            while not self._stop.is_set() and client.is_connected:
                await asyncio.sleep(0.2)
            try:
                await client.stop_notify(BLE_DATA_CHAR_UUID)
            except Exception:
                pass

    def send(self, text):
        if self.connected and self.client and self.loop:
            fut = asyncio.run_coroutine_threadsafe(
                self.client.write_gatt_char(BLE_CMD_CHAR_UUID, text.encode()), self.loop
            )
            try:
                fut.result(timeout=3)
            except Exception:
                pass

    def disconnect(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=4)
        self.thread = None
        self._connected = False
        self.client = None


def open_in_os(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception as e:
        messagebox.showwarning("Couldn't open file", str(e))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gesture Recognizer Control Panel")
        self.geometry("880x620")

        self.settings = load_settings()
        self.link_mode = tk.StringVar(value=self.settings.get("link_mode", "USB"))
        self.serial = SerialLink()  # currently active link - USB or BLE, same interface either way
        self.gesture_names = []

        self.dataset_path = tk.StringVar(value=self.settings["dataset_path"])
        self.dataset_file = None
        self.collect_counts = {}

        self.training_proc = None
        self.training_queue = queue.Queue()

        self.run_logging = False
        self.run_rows = []

        self._build_connection_bar()
        self._build_tabs()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._try_load_gestures_on_start()
        self.after(50, self._poll_serial)

    # -- connection bar, shared by Collect and Run tabs -------------------
    def _build_connection_bar(self):
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")

        ttk.Radiobutton(bar, text="USB", variable=self.link_mode, value="USB",
                        command=self._on_mode_change).pack(side="left")
        ttk.Radiobutton(bar, text="BLE", variable=self.link_mode, value="BLE",
                        command=self._on_mode_change).pack(side="left", padx=(0, 8))

        # USB controls
        self.usb_frame = ttk.Frame(bar)
        ttk.Label(self.usb_frame, text="Port:").pack(side="left")
        self.port_var = tk.StringVar(value=self.settings.get("last_port", ""))
        self.port_combo = ttk.Combobox(self.usb_frame, textvariable=self.port_var, width=18, state="readonly")
        self.port_combo.pack(side="left", padx=4)
        ttk.Button(self.usb_frame, text="Refresh", command=self._refresh_ports).pack(side="left")

        # BLE controls
        self.ble_frame = ttk.Frame(bar)
        ttk.Label(self.ble_frame, text="Device name:").pack(side="left")
        self.ble_name_var = tk.StringVar(value=self.settings.get("ble_name", BLE_DEVICE_NAME))
        ttk.Entry(self.ble_frame, textvariable=self.ble_name_var, width=20).pack(side="left", padx=4)
        if not BLEAK_AVAILABLE:
            ttk.Label(self.ble_frame, text="(needs: pip install bleak)", foreground="#a33").pack(side="left")

        self.connect_btn = ttk.Button(bar, text="Connect", command=self._toggle_connect)
        self.status_var = tk.StringVar(value="disconnected")
        self.status_label = ttk.Label(bar, textvariable=self.status_var, foreground="#555")

        self._refresh_ports()
        self._on_mode_change()  # show the right sub-frame for the saved mode

    def _on_mode_change(self):
        if self.serial.connected:
            self.serial.disconnect()
            self.connect_btn.config(text="Connect")
            self.status_var.set("disconnected")
        self.usb_frame.pack_forget()
        self.ble_frame.pack_forget()
        self.connect_btn.pack_forget()
        self.status_label.pack_forget()
        if self.link_mode.get() == "USB":
            self.usb_frame.pack(side="left")
        else:
            self.ble_frame.pack(side="left")
        self.connect_btn.pack(side="left", padx=8)
        self.status_label.pack(side="left", padx=8)
        self.settings["link_mode"] = self.link_mode.get()
        save_settings(self.settings)

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if self.port_var.get() not in ports and ports:
            self.port_var.set(ports[0])

    def _toggle_connect(self):
        if self.serial.connected:
            self.serial.disconnect()
            self.connect_btn.config(text="Connect")
            self.status_var.set("disconnected")
            return

        if self.link_mode.get() == "BLE":
            if not BLEAK_AVAILABLE:
                messagebox.showerror("bleak not installed", "Run: pip install bleak")
                return
            name = self.ble_name_var.get().strip()
            if not name:
                messagebox.showwarning("No device name", "Enter the BLE device name (default: GestureRecognizer).")
                return
            self.serial = BLELink()
            self.status_var.set(f"scanning for {name}...")
            self.update_idletasks()
            self.serial.connect(name)
            self.settings["ble_name"] = name
            # BLE connects in the background; the status label updates via
            # the usual _dispatch_line("# BLE connected") message, same as
            # any other status line. Give the button a sensible default now.
            self.connect_btn.config(text="Disconnect")
            save_settings(self.settings)
        else:
            port = self.port_var.get()
            if not port:
                messagebox.showwarning("No port", "Pick a serial port first.")
                return
            self.serial = SerialLink()
            try:
                self.serial.connect(port)
            except Exception as e:
                messagebox.showerror("Connection failed", str(e))
                return
            self.connect_btn.config(text="Disconnect")
            self.status_var.set(f"connected: {port}")
            self.settings["last_port"] = port
            save_settings(self.settings)

    # -- tabs ---------------------------------------------------------------
    def _build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._build_gestures_tab(nb)
        self._build_collect_tab(nb)
        self._build_train_tab(nb)
        self._build_run_tab(nb)

    # ===== Gestures tab =====================================================
    def _build_gestures_tab(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Gestures")

        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Label(top, text="gesture_config.h:").pack(side="left")
        self.gcfg_path_var = tk.StringVar(value=self.settings.get("gesture_config_path", ""))
        ttk.Entry(top, textvariable=self.gcfg_path_var, width=60).pack(side="left", padx=4)
        ttk.Button(top, text="Browse...", command=self._browse_gesture_config).pack(side="left")
        ttk.Button(top, text="Reload", command=self._load_gestures_from_path).pack(side="left", padx=4)

        mid = ttk.Frame(tab)
        mid.pack(fill="both", expand=True, pady=8)
        self.gesture_listbox = tk.Listbox(mid, height=12)
        self.gesture_listbox.pack(side="left", fill="both", expand=True)
        btns = ttk.Frame(mid)
        btns.pack(side="left", fill="y", padx=8)
        self.new_gesture_var = tk.StringVar()
        ttk.Entry(btns, textvariable=self.new_gesture_var, width=18).pack(pady=(0, 4))
        ttk.Button(btns, text="Add", command=self._add_gesture).pack(fill="x")
        ttk.Button(btns, text="Remove selected", command=self._remove_gesture).pack(fill="x", pady=4)
        ttk.Button(btns, text="Save to file", command=self._save_gestures).pack(fill="x", pady=(20, 0))

        ttk.Label(
            tab,
            text="After saving, re-upload 01_data_collection.ino (and 03_inference.ino, if it's\n"
                 "flashed) so the board picks up the new list. This only edits the on-device\n"
                 "label menu - a gesture isn't actually recognizable until you've collected\n"
                 "data for it and retrained.",
            foreground="#555", justify="left",
        ).pack(anchor="w", pady=(8, 0))

    def _browse_gesture_config(self):
        path = filedialog.askopenfilename(
            title="Select gesture_config.h",
            filetypes=[("Header files", "*.h"), ("All files", "*.*")],
        )
        if path:
            self.gcfg_path_var.set(path)
            self.settings["gesture_config_path"] = path
            save_settings(self.settings)
            self._load_gestures_from_path()

    def _try_load_gestures_on_start(self):
        if self.gcfg_path_var.get():
            try:
                self._load_gestures_from_path()
            except Exception:
                pass

    def _load_gestures_from_path(self):
        path = self.gcfg_path_var.get()
        if not path:
            return
        try:
            labels = read_gesture_labels(path)
        except Exception as e:
            messagebox.showerror("Couldn't read file", str(e))
            return
        self.gesture_names = labels
        self.gesture_listbox.delete(0, tk.END)
        for name in labels:
            self.gesture_listbox.insert(tk.END, name)
        self._refresh_gesture_dependents()

    def _add_gesture(self):
        name = self.new_gesture_var.get().strip()
        if not name:
            return
        if name in self.gesture_listbox.get(0, tk.END):
            messagebox.showwarning("Already exists", f'"{name}" is already in the list.')
            return
        self.gesture_listbox.insert(tk.END, name)
        self.new_gesture_var.set("")

    def _remove_gesture(self):
        sel = self.gesture_listbox.curselection()
        if sel:
            self.gesture_listbox.delete(sel[0])

    def _save_gestures(self):
        path = self.gcfg_path_var.get()
        if not path:
            messagebox.showwarning("No file", "Browse to gesture_config.h first.")
            return
        labels = list(self.gesture_listbox.get(0, tk.END))
        if not labels:
            messagebox.showwarning("Empty list", "Add at least one gesture first.")
            return
        try:
            write_gesture_labels(path, labels)
        except Exception as e:
            messagebox.showerror("Couldn't save", str(e))
            return
        self.gesture_names = labels
        self._refresh_gesture_dependents()
        messagebox.showinfo("Saved", "gesture_config.h updated. Re-upload your sketches to apply it on-device.")

    def _refresh_gesture_dependents(self):
        self.collect_gesture_combo["values"] = self.gesture_names
        if self.gesture_names and not self.collect_gesture_var.get():
            self.collect_gesture_var.set(self.gesture_names[0])
        for name in self.gesture_names:
            self.collect_counts.setdefault(name, 0)
        self._refresh_counts_view()

    # ===== Collect tab =======================================================
    def _build_collect_tab(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Collect")

        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="Dataset CSV:").pack(side="left")
        ttk.Entry(row, textvariable=self.dataset_path, width=50).pack(side="left", padx=4)
        ttk.Button(row, text="Browse...", command=self._browse_dataset).pack(side="left")
        ttk.Button(row, text="Open folder", command=lambda: open_in_os(Path(self.dataset_path.get()).parent)).pack(
            side="left", padx=4
        )

        row2 = ttk.Frame(tab)
        row2.pack(fill="x", pady=8)
        ttk.Label(row2, text="Label:").pack(side="left")
        self.collect_gesture_var = tk.StringVar()
        self.collect_gesture_combo = ttk.Combobox(
            row2, textvariable=self.collect_gesture_var, values=self.gesture_names, state="readonly", width=20
        )
        self.collect_gesture_combo.pack(side="left", padx=4)
        self.collect_gesture_combo.bind("<<ComboboxSelected>>", self._on_collect_label_change)
        self.collect_start_btn = ttk.Button(row2, text="Start streaming", command=self._start_collecting)
        self.collect_start_btn.pack(side="left", padx=8)
        self.collect_stop_btn = ttk.Button(row2, text="Stop", command=self._stop_collecting, state="disabled")
        self.collect_stop_btn.pack(side="left")

        ttk.Label(tab, text="Samples collected this session:").pack(anchor="w", pady=(8, 0))
        self.counts_tree = ttk.Treeview(tab, columns=("gesture", "count"), show="headings", height=8)
        self.counts_tree.heading("gesture", text="gesture")
        self.counts_tree.heading("count", text="count")
        self.counts_tree.column("gesture", width=200)
        self.counts_tree.column("count", width=100)
        self.counts_tree.pack(fill="both", expand=True)

        ttk.Label(
            tab,
            text="Connect above, pick a label, hit Start, hold the gesture (vary it slightly), then Stop.\n"
                 "Repeat for every gesture, ideally across more than one sitting.",
            foreground="#555",
        ).pack(anchor="w", pady=(8, 0))

    def _browse_dataset(self):
        path = filedialog.asksaveasfilename(
            title="Dataset CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile=Path(self.dataset_path.get()).name,
        )
        if path:
            self.dataset_path.set(path)
            self.settings["dataset_path"] = path
            save_settings(self.settings)
            if self.dataset_file:
                self.dataset_file.close()
                self.dataset_file = None

    def _on_collect_label_change(self, _evt=None):
        if self.gesture_names and self.collect_gesture_var.get() in self.gesture_names:
            idx = self.gesture_names.index(self.collect_gesture_var.get())
            self.serial.send(f"l{idx}")

    def _start_collecting(self):
        if not self.serial.connected:
            messagebox.showwarning("Not connected", "Connect to the board first.")
            return
        self._on_collect_label_change()
        self.serial.send("s")
        self.collect_start_btn.config(state="disabled")
        self.collect_stop_btn.config(state="normal")

    def _stop_collecting(self):
        self.serial.send("x")
        self.collect_start_btn.config(state="normal")
        self.collect_stop_btn.config(state="disabled")

    def _ensure_dataset_file(self):
        path = self.dataset_path.get()
        if not path:
            return None
        if self.dataset_file is None or self.dataset_file.name != path:
            if self.dataset_file:
                self.dataset_file.close()
            self.dataset_file = open(path, "a", newline="")
        return self.dataset_file

    def _handle_collect_line(self, parts):
        label = parts[0]
        f = self._ensure_dataset_file()
        if f:
            f.write(",".join(parts) + "\n")
            f.flush()
        self.collect_counts[label] = self.collect_counts.get(label, 0) + 1
        self._refresh_counts_view()

    def _refresh_counts_view(self):
        self.counts_tree.delete(*self.counts_tree.get_children())
        for name in self.gesture_names:
            self.counts_tree.insert("", tk.END, values=(name, self.collect_counts.get(name, 0)))

    # ===== Train tab ==========================================================
    def _build_train_tab(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Train")

        steps = (
            "How to tune it:\n"
            "1. Collect data for every gesture (Collect tab) - a few hundred samples each is a good start.\n"
            "2. Point \"Dataset CSV\" below at that file.\n"
            "3. Leave hidden layers / epochs / learning rate at the defaults for a first run.\n"
            "4. Click Train and watch the log. Want: train_acc and val_acc both climbing toward 1.0.\n"
            "   If val_acc lags well behind train_acc, that's overfitting - collect more data,\n"
            "   or try smaller hidden layers.\n"
            "5. If it looks good, give it a name and click \"Save as named training\" - it's kept below\n"
            "   so you can come back to it later without retraining.\n"
            "6. Click Deploy to copy it into your Arduino library, then re-upload 03_inference.ino."
        )
        ttk.Label(tab, text=steps, justify="left", foreground="#333").pack(anchor="w", pady=(0, 8))

        form = ttk.Frame(tab)
        form.pack(fill="x")
        ttk.Label(form, text="Dataset CSV:").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.dataset_path, width=45).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(form, text="Browse...", command=self._browse_dataset).grid(row=0, column=2)

        ttk.Label(form, text="Hidden layers:").grid(row=1, column=0, sticky="w", pady=4)
        self.hidden_var = tk.StringVar(value="16 12")
        ttk.Entry(form, textvariable=self.hidden_var, width=20).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(form, text="Epochs:").grid(row=2, column=0, sticky="w")
        self.epochs_var = tk.StringVar(value="300")
        ttk.Entry(form, textvariable=self.epochs_var, width=10).grid(row=2, column=1, sticky="w", padx=4)

        ttk.Label(form, text="Learning rate:").grid(row=3, column=0, sticky="w")
        self.lr_var = tk.StringVar(value="0.01")
        ttk.Entry(form, textvariable=self.lr_var, width=10).grid(row=3, column=1, sticky="w", padx=4)

        self.train_btn = ttk.Button(tab, text="Train", command=self._start_training)
        self.train_btn.pack(anchor="w", pady=6)

        self.train_log = tk.Text(tab, height=10, state="disabled", bg="#111", fg="#ddd")
        self.train_log.pack(fill="both", expand=True)

        save_row = ttk.Frame(tab)
        save_row.pack(fill="x", pady=6)
        ttk.Label(save_row, text="Name this training:").pack(side="left")
        self.train_name_var = tk.StringVar()
        ttk.Entry(save_row, textvariable=self.train_name_var, width=24).pack(side="left", padx=4)
        ttk.Button(save_row, text="Save as named training", command=self._save_named_training).pack(side="left")

        ttk.Label(tab, text="Saved trainings:").pack(anchor="w", pady=(8, 0))
        self.saved_tree = ttk.Treeview(tab, columns=("acc", "when"), show="tree headings", height=6)
        self.saved_tree.heading("#0", text="name")
        self.saved_tree.heading("acc", text="val_acc")
        self.saved_tree.heading("when", text="saved")
        self.saved_tree.pack(fill="both", expand=True)
        actions = ttk.Frame(tab)
        actions.pack(fill="x", pady=4)
        ttk.Button(actions, text="Deploy selected", command=self._deploy_training).pack(side="left")
        ttk.Button(actions, text="Delete selected", command=self._delete_training).pack(side="left", padx=4)
        self._refresh_saved_trainings()

    def _log_train(self, text):
        self.train_log.config(state="normal")
        self.train_log.insert(tk.END, text + "\n")
        self.train_log.see(tk.END)
        self.train_log.config(state="disabled")

    def _start_training(self):
        if self.training_proc is not None:
            messagebox.showinfo("Already running", "A training run is already in progress.")
            return
        data_path = self.dataset_path.get()
        if not Path(data_path).exists():
            messagebox.showwarning("No dataset", f"Can't find {data_path}")
            return
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--data", data_path,
            "--hidden", *self.hidden_var.get().split(),
            "--epochs", self.epochs_var.get(),
            "--lr", self.lr_var.get(),
            "--out", str(CURRENT_MODEL_PATH),
        ]
        self.train_log.config(state="normal")
        self.train_log.delete("1.0", tk.END)
        self.train_log.config(state="disabled")
        self._log_train("$ " + " ".join(cmd))
        self.train_btn.config(state="disabled")

        def run():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                self.training_proc = proc
                for line in proc.stdout:
                    self.training_queue.put(line.rstrip())
                proc.wait()
                self.training_queue.put(("__DONE__", proc.returncode))
            except Exception as e:
                self.training_queue.put(f"ERROR: {e}")
                self.training_queue.put(("__DONE__", 1))

        threading.Thread(target=run, daemon=True).start()
        self.after(50, self._poll_training)

    def _poll_training(self):
        done = False
        try:
            while True:
                item = self.training_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__DONE__":
                    done = True
                    rc = item[1]
                else:
                    self._log_train(item)
        except queue.Empty:
            pass
        if done:
            self.training_proc = None
            self.train_btn.config(state="normal")
            self._log_train("--- finished ---" if rc == 0 else f"--- exited with code {rc} ---")
            return
        self.after(100, self._poll_training)

    def _last_val_acc(self):
        text = self.train_log.get("1.0", tk.END)
        matches = re.findall(r"val_acc\s+([\d.]+|nan)", text)
        return matches[-1] if matches else "?"

    def _save_named_training(self):
        name = self.train_name_var.get().strip()
        if not name:
            messagebox.showwarning("No name", "Give this training a name first.")
            return
        if not CURRENT_MODEL_PATH.exists():
            messagebox.showwarning("Nothing to save", "Run a training first.")
            return
        dest_dir = SAVED_TRAININGS_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(CURRENT_MODEL_PATH, dest_dir / "model_weights.h")
        meta = {
            "name": name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "hidden": self.hidden_var.get(),
            "epochs": self.epochs_var.get(),
            "lr": self.lr_var.get(),
            "dataset": self.dataset_path.get(),
            "val_acc": self._last_val_acc(),
            "gestures": self.gesture_names,
        }
        (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self.train_name_var.set("")
        self._refresh_saved_trainings()
        messagebox.showinfo("Saved", f'Training "{name}" saved - you can reload or deploy it anytime.')

    def _refresh_saved_trainings(self):
        self.saved_tree.delete(*self.saved_tree.get_children())
        if not SAVED_TRAININGS_DIR.exists():
            return
        for d in sorted(SAVED_TRAININGS_DIR.iterdir()):
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            self.saved_tree.insert("", tk.END, iid=str(d), text=meta.get("name", d.name),
                                    values=(meta.get("val_acc", "?"), meta.get("saved_at", "?")))

    def _selected_training_dir(self):
        sel = self.saved_tree.selection()
        if not sel:
            messagebox.showwarning("Nothing selected", "Pick a saved training first.")
            return None
        return Path(sel[0])

    def _deploy_training(self):
        d = self._selected_training_dir()
        if not d:
            return
        dest = self.settings.get("library_model_path", "")
        if not dest:
            dest = filedialog.asksaveasfilename(
                title="Select GestureModel/src/model_weights.h destination",
                initialfile="model_weights.h",
                filetypes=[("Header files", "*.h")],
            )
            if not dest:
                return
            self.settings["library_model_path"] = dest
            save_settings(self.settings)
        shutil.copy(d / "model_weights.h", dest)
        messagebox.showinfo("Deployed", f"Copied to {dest}.\nRe-upload 03_inference.ino to use it.")

    def _delete_training(self):
        d = self._selected_training_dir()
        if not d:
            return
        if messagebox.askyesno("Delete", f"Delete saved training \"{d.name}\"? This can't be undone."):
            shutil.rmtree(d, ignore_errors=True)
            self._refresh_saved_trainings()

    # ===== Run tab ============================================================
    def _build_run_tab(self, nb):
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Run")

        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="Polling rate (Hz):").pack(side="left")
        self.poll_hz_var = tk.StringVar(value="50")
        ttk.Spinbox(row, from_=1, to=200, textvariable=self.poll_hz_var, width=6).pack(side="left", padx=4)
        ttk.Button(row, text="Apply", command=self._apply_poll_hz).pack(side="left")

        row2 = ttk.Frame(tab)
        row2.pack(fill="x", pady=6)
        ttk.Label(row2, text="Gesture meaning:").pack(side="left")
        self.mapping_var = tk.StringVar(value="(raw gesture names)")
        self.mapping_combo = ttk.Combobox(row2, textvariable=self.mapping_var, state="readonly", width=24)
        self.mapping_combo.pack(side="left", padx=4)
        ttk.Button(row2, text="Edit mappings...", command=self._edit_mapping).pack(side="left")
        self._refresh_mapping_list()

        row3 = ttk.Frame(tab)
        row3.pack(fill="x", pady=6)
        self.run_start_btn = ttk.Button(row3, text="Start logging", command=self._start_run_logging)
        self.run_start_btn.pack(side="left")
        self.run_stop_btn = ttk.Button(row3, text="Stop logging", command=self._stop_run_logging, state="disabled")
        self.run_stop_btn.pack(side="left", padx=4)
        ttk.Button(row3, text="Clear", command=self._clear_run_log).pack(side="left", padx=4)
        ttk.Button(row3, text="Export CSV...", command=self._export_run_log).pack(side="left", padx=12)

        self.run_tree = ttk.Treeview(
            tab, columns=("time", "gesture", "meaning", "confidence"), show="headings", height=16
        )
        for col, w in (("time", 110), ("gesture", 140), ("meaning", 160), ("confidence", 90)):
            self.run_tree.heading(col, text=col)
            self.run_tree.column(col, width=w)
        self.run_tree.pack(fill="both", expand=True)

    def _apply_poll_hz(self):
        if not self.serial.connected:
            messagebox.showwarning("Not connected", "Connect to the board first.")
            return
        try:
            hz = int(self.poll_hz_var.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Polling rate must be a whole number.")
            return
        self.serial.send(f"r{hz}")

    def _refresh_mapping_list(self):
        names = ["(raw gesture names)"] + sorted(self.settings.get("mappings", {}).keys())
        self.mapping_combo["values"] = names
        if self.mapping_var.get() not in names:
            self.mapping_var.set(names[0])

    def _edit_mapping(self):
        if not self.gesture_names:
            messagebox.showwarning("No gestures loaded", "Load gesture_config.h in the Gestures tab first.")
            return
        dlg = tk.Toplevel(self)
        dlg.title("Edit gesture meaning mapping")
        entries = {}
        for i, name in enumerate(self.gesture_names):
            ttk.Label(dlg, text=name).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            var = tk.StringVar(value=self.settings.get("mappings", {}).get(self.mapping_var.get(), {}).get(name, name))
            ttk.Entry(dlg, textvariable=var, width=30).grid(row=i, column=1, padx=6, pady=2)
            entries[name] = var

        name_row = len(self.gesture_names)
        ttk.Label(dlg, text="Profile name:").grid(row=name_row, column=0, sticky="w", padx=6, pady=(10, 2))
        profile_name_var = tk.StringVar(
            value="" if self.mapping_var.get() == "(raw gesture names)" else self.mapping_var.get()
        )
        ttk.Entry(dlg, textvariable=profile_name_var, width=30).grid(row=name_row, column=1, padx=6, pady=(10, 2))

        def save_and_close():
            pname = profile_name_var.get().strip()
            if not pname:
                messagebox.showwarning("No name", "Give this mapping profile a name.", parent=dlg)
                return
            self.settings.setdefault("mappings", {})[pname] = {k: v.get() for k, v in entries.items()}
            save_settings(self.settings)
            self._refresh_mapping_list()
            self.mapping_var.set(pname)
            dlg.destroy()

        ttk.Button(dlg, text="Save profile", command=save_and_close).grid(
            row=name_row + 1, column=0, columnspan=2, pady=10
        )

    def _start_run_logging(self):
        if not self.serial.connected:
            messagebox.showwarning("Not connected", "Connect to the board first.")
            return
        self.run_logging = True
        self.run_start_btn.config(state="disabled")
        self.run_stop_btn.config(state="normal")

    def _stop_run_logging(self):
        self.run_logging = False
        self.run_start_btn.config(state="normal")
        self.run_stop_btn.config(state="disabled")

    def _clear_run_log(self):
        self.run_rows.clear()
        self.run_tree.delete(*self.run_tree.get_children())

    def _handle_run_line(self, parts):
        if not self.run_logging:
            return
        try:
            gesture, confidence = parts[0], float(parts[1])
        except ValueError:
            return
        profile = self.mapping_var.get()
        meaning = gesture
        if profile != "(raw gesture names)":
            meaning = self.settings.get("mappings", {}).get(profile, {}).get(gesture, gesture)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        row = (ts, gesture, meaning, f"{confidence:.2f}")
        self.run_rows.append(row)
        self.run_tree.insert("", tk.END, values=row)
        self.run_tree.see(self.run_tree.get_children()[-1])

    def _export_run_log(self):
        if not self.run_rows:
            messagebox.showinfo("Nothing to export", "No logged rows yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Export session log", defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "gesture", "meaning", "confidence"])
            w.writerows(self.run_rows)
        if messagebox.askyesno("Exported", f"Saved to {path}.\nOpen it now?"):
            open_in_os(path)

    # -- central serial dispatch --------------------------------------------
    def _poll_serial(self):
        try:
            while True:
                line = self.serial.q.get_nowait()
                self._dispatch_line(line)
        except queue.Empty:
            pass
        self.after(50, self._poll_serial)

    def _dispatch_line(self, line):
        self.status_var.set(line[:60])
        if line.startswith("#"):
            return
        parts = line.split(",")
        if len(parts) == 6:
            self._handle_collect_line(parts)
        elif len(parts) == 2:
            self._handle_run_line(parts)

    def _on_close(self):
        self.serial.disconnect()
        if self.dataset_file:
            self.dataset_file.close()
        self.settings["dataset_path"] = self.dataset_path.get()
        self.settings["gesture_config_path"] = self.gcfg_path_var.get()
        save_settings(self.settings)
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
