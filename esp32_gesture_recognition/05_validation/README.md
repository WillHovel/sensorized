# Glove validation: three commands

Needs: `pip install numpy scikit-learn matplotlib pyserial`

## 1. Accuracy + confusion matrix (A, B): about 1-3 minutes, no hardware
    python validate_glove.py            # add --fast while iterating
Reads `data/*.csv`, or `../04_gui/dataset.csv` if `data/` is empty (then it uses the weaker
blocked-temporal fallback and says so).

**Collecting data so the result is honest:** one CSV per sitting, named `person__session.csv`
(`will__2026-10-20_a.csv`, `anna__01.csv`). In the GUI Collect tab, click Browse... and pick a new
file each time you put the glove on. Per sitting: each gesture ~20 s, a short relaxed-hand gap
between, 2 rounds (about 5 minutes). Lean target: 4+ sitting of yours on 2+ days, plus 1-2 sittings
from 2 friends. 4+ sessions supports "held-out sessions"; 2+ people also gives "unseen person".
Optional flags: `--augment 0.05` (strap-shift jitter), `--session-norm` (simulated calibration).
Use `--lr` equal to the learning rate you actually deploy with (your GUI run used 0.1).

## 2. On-device numbers (E, F, G): about 3 minutes
Flash `06_bench/06_bench.ino` (same libraries as 03_inference), close the serial monitor, then:
    python bench_capture.py --port COM5            # add --skip-e for a 15 s run
Prompts you to swap in a 10k 1% resistor, then a flex sensor taped flat, for the ADC-noise test.

## 3. Report
    python make_report.py --compile-log compile.txt   # compile.txt = pasted Arduino build output (optional, adds flash/RAM)
Writes `results/report.md`, `results/website_snippet.html`, and the PNG figures. Copy the figures and
snippet to the site; use the bullet fragments on the resume.

Not covered on purpose: majority vote/confidence gate (C), false triggers (D), sensor drift (H).
Latency here is firmware-only; finger-to-pixel latency still needs a 240 fps video.
