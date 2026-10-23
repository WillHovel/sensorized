#!/usr/bin/env python3
"""
Measurements E, F, G: run the 06_bench sketch and capture everything in one go.

    python bench_capture.py --port COM5                 # F + G + E (2 conditions, 30 s each)
    python bench_capture.py --port COM5 --skip-e        # ~15 s: inference time + latency only
    python bench_capture.py --port /dev/ttyACM0 --channel 2 --seconds 60

Only E needs your hands: it asks you to swap the input between a fixed 10k 1%
resistor (electronics noise) and a flex sensor taped flat (adds sensor noise).
Output: results/bench.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "results"


def run(ser, cmd, timeout):
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    lines, end = [], time.time() + timeout
    while time.time() < end:
        raw = ser.readline().decode(errors="ignore").strip()
        if raw == "#done":
            return lines
        if raw and not raw.startswith("#"):
            lines.append(raw)
    sys.exit(f"Timed out waiting for reply to '{cmd}' (is 06_bench flashed?)")


def adc_stats(rows):
    reads = np.array([[float(v) for v in r.split(",")[2:10]] for r in rows if len(r.split(",")) == 10])
    one, avg = reads[:, 0], reads.mean(axis=1)

    def d(x):
        sd = max(float(x.std()), 1e-3)
        return {"sd_lsb": sd, "pp_lsb": float(x.max() - x.min()),
                "noise_free_bits": float(np.log2(4096 / (6.6 * sd)))}
    a, b = d(one), d(avg)
    return {"ticks": int(len(reads)), "mean_counts": float(one.mean()), "one_read": a, "oversampled": b,
            "improvement_x": a["sd_lsb"] / b["sd_lsb"], "ideal_x": float(np.sqrt(8))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--skip-e", action="store_true")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--conditions", nargs="+", default=["fixed_resistor", "sensor_still"])
    a = ap.parse_args()
    import serial
    ser = serial.Serial(a.port, a.baud, timeout=1)
    time.sleep(2.5)  # board resets when the port opens
    res = {}

    info = {}
    for l in run(ser, "i", 5):
        _, k, v = l.split(",")
        info[k] = float(v)
    res["info"] = info
    print("info:", info)

    f = run(ser, "f", 30)[0].split(",")
    res["inference_us"] = {"min": float(f[1]), "max": float(f[2]), "mean": float(f[3])}
    res["cpu_load_pct_at_50hz"] = 100 * res["inference_us"]["mean"] / 20000
    print("inference:", res["inference_us"])

    g = np.array([[float(v) for v in l.split(",")[1:]] for l in run(ser, "g", 30)])
    tot = g.sum(axis=1)
    res["firmware_latency_us"] = {
        "n": int(len(g)), "adc_median": float(np.median(g[:, 0])), "infer_median": float(np.median(g[:, 1])),
        "oled_median": float(np.median(g[:, 2])), "total_median": float(np.median(tot)),
        "total_p95": float(np.percentile(tot, 95)), "total_max": float(tot.max())}
    print("latency:", res["firmware_latency_us"])

    if not a.skip_e:
        res["adc_noise"] = {}
        ticks = a.seconds * 50
        for cond in a.conditions:
            input(f"\n[E] Set up condition '{cond}' on channel {a.channel}, then press Enter...")
            rows = run(ser, f"e{a.channel},{ticks}", a.seconds + 30)
            res["adc_noise"][cond] = adc_stats(rows)
            r = res["adc_noise"][cond]
            print(f"  {cond}: {r['one_read']['sd_lsb']:.2f} -> {r['oversampled']['sd_lsb']:.2f} LSB rms "
                  f"({r['improvement_x']:.2f}x vs 2.83x ideal)")
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / "bench.json", "w"), indent=2)
    print("wrote", OUT / "bench.json")


if __name__ == "__main__":
    main()
