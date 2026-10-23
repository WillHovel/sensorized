#!/usr/bin/env python3
"""
Turn results/validation.json + results/bench.json into publishable text.

    python make_report.py
    python make_report.py --compile-log compile.txt     # paste of the Arduino 'Sketch uses...' output of 03_inference

Writes results/report.md (table, method, resume bullets) and results/website_snippet.html.
Only measured numbers are printed; anything not measured is simply omitted, never guessed.
"""
import argparse, json, re
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-log", default=None)
    a = ap.parse_args()
    v, b = load("validation.json"), load("bench.json")
    if not v and not b:
        raise SystemExit("Run validate_glove.py and/or bench_capture.py first.")
    rows, bullets, warns, method = [], [], [], []
    pct = lambda x: f"{100 * x:.1f}%"

    if v:
        s = v["held_out_sessions"]
        proxy = v["mode"].startswith("blocked")
        if proxy:
            warns.append("Accuracy used the single-file blocked-temporal fallback (no real sessions/people). "
                         "It is better than a random split but NOT a re-worn-glove or new-user result. "
                         "Collect per-sitting files (person__session.csv) before publishing a headline number.")
        elif v["n_people"] < 2:
            warns.append("Only one person in the data: report held-out SESSIONS only, not new-user accuracy.")
        label = ("blocked temporal CV (proxy)" if proxy else
                 f"leave-one-session-out, {v['n_sessions']} sessions, {v['n_people']} "
                 f"{'person' if v['n_people'] == 1 else 'people'}")
        rows.append(("Held-out accuracy", f"{pct(s['mean'])} ± {100*s['std']:.1f} (worst fold {pct(s['worst'])})", label))
        if "held_out_people" in v:
            p = v["held_out_people"]
            rows.append(("Unseen-person accuracy", f"{pct(p['mean'])} ± {100*p['std']:.1f}",
                         f"leave-one-person-out, {v['n_people']} people"))
        rows.append(("Random 80/20 split (reference)", pct(v["random_split"]["mean"]),
                     "overstates real performance: neighbouring samples leak across the split"))
        bc = v["biggest_confusion"]
        rows.append(("Most-confused pair", f"{bc['true']} → {bc['pred']} ({100*bc['rate']:.1f}%)", "pooled held-out folds"))
        arch = f"{'→'.join(str(x) for x in [5, *v['settings']['hidden'], len(v['classes'])])}"
        method.append(f"Accuracy: Python replay of the firmware pipeline (per-sample argmax, min-max constants "
                      f"from training folds only, inputs clipped to [0,1]), {v['n_samples']} samples, "
                      f"{len(v['classes'])} gestures; majority vote and confidence gate are not included.")
        bullets.append(("acc", f"{pct(s['mean'])} accuracy on held-out data ({label})"))

    if b:
        i = b["info"]
        us = b["inference_us"]
        params = int(i["params"])
        rows.append(("Inference time", f"{us['mean']:.0f} µs ({us['min']:.0f}–{us['max']:.0f}), "
                     f"{b['cpu_load_pct_at_50hz']:.2f}% CPU at 50 Hz",
                     f"on-device, {int(i['cpu_mhz'])} MHz, {params} parameters, {int(i['macs'])} MACs, 5×1000 runs"))
        rows.append(("Free heap (bench sketch)", f"{int(i['heap_free'])//1024} KB of {int(i['heap_total'])//1024} KB",
                     "on-device; model weights are const (flash)"))
        l = b["firmware_latency_us"]
        rows.append(("Firmware latency", f"{l['total_median']/1000:.1f} ms median, {l['total_p95']/1000:.1f} ms p95",
                     f"ADC {l['adc_median']/1000:.1f} + inference {l['infer_median']/1000:.2f} + OLED "
                     f"{l['oled_median']/1000:.1f} ms per cycle, {l['n']} cycles"))
        method.append("Latency is the firmware pipeline only (ADC read → OLED update returns); it excludes the "
                      "sampling period and the vote delay, and is not finger-to-pixel latency.")
        for cond, r in b.get("adc_noise", {}).items():
            rows.append((f"ADC noise ({cond.replace('_', ' ')})",
                         f"{r['one_read']['sd_lsb']:.2f} → {r['oversampled']['sd_lsb']:.2f} LSB rms "
                         f"({r['improvement_x']:.2f}× vs {r['ideal_x']:.2f}× ideal)",
                         f"noise-free bits {r['one_read']['noise_free_bits']:.1f} → "
                         f"{r['oversampled']['noise_free_bits']:.1f}; {r['ticks']} ticks, 8× oversampling"))
        bullets.append(("hw", f"{us['mean']:.0f} µs per inference on ESP32-S3 ({params}-parameter MLP), "
                              f"{l['total_median']/1000:.0f} ms median firmware latency"))
        if "adc_noise" in b and b["adc_noise"]:
            r = list(b["adc_noise"].values())[-1]
            bullets.append(("adc", f"8× oversampling cut ADC noise {r['improvement_x']:.1f}× "
                                   f"({r['one_read']['sd_lsb']:.1f} → {r['oversampled']['sd_lsb']:.1f} LSB rms)"))

    if a.compile_log:
        t = open(a.compile_log, errors="ignore").read()
        m1 = re.search(r"Sketch uses (\d+) bytes \((\d+)%\) of program storage space. Maximum is (\d+)", t)
        m2 = re.search(r"Global variables use (\d+) bytes \((\d+)%\) of dynamic memory", t)
        if m1:
            rows.append(("Flash (03_inference build)", f"{int(m1.group(1))/1024:.0f} KB ({m1.group(2)}%)", "Arduino compile output, includes BLE + display"))
        if m2:
            rows.append(("Static RAM (03_inference build)", f"{int(m2.group(1))/1024:.1f} KB ({m2.group(2)}%)", "Arduino compile output"))

    md = ["# Motion Sense Glove: validation results\n", "| Metric | Result | How measured |", "|---|---|---|"]
    md += [f"| {m} | {r} | {h} |" for m, r, h in rows]
    md += ["", "## Method notes"] + [f"- {m}" for m in method]
    md += ["", "## Resume bullet fragments (measured numbers only)"] + [f"- {t}" for _, t in bullets]
    if warns:
        md += ["", "## Before you publish"] + [f"- WARNING: {w}" for w in warns]
    (R / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    h = ["<section id=\"validation\">", "<h2>Validation</h2>",
         "<p>Every number below is regenerated by scripts in the repo: held-out cross-validation in Python, "
         "plus a benchmark sketch that measures inference time, latency and ADC noise on the board itself.</p>",
         "<table>", "<tr><th>Metric</th><th>Result</th><th>How measured</th></tr>"]
    h += [f"<tr><td>{m}</td><td>{r}</td><td>{x}</td></tr>" for m, r, x in rows]
    h += ["</table>"]
    if (R / "confusion_matrix.png").exists():
        h += ['<img src="confusion_matrix.png" alt="Held-out confusion matrix" width="480">',
              '<img src="accuracy_comparison.png" alt="Random split vs held-out accuracy" width="360">']
    h += ["<ul>"] + [f"<li>{m}</li>" for m in method] + ["</ul>", "</section>"]
    (R / "website_snippet.html").write_text("\n".join(h) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
