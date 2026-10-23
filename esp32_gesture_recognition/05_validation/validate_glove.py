#!/usr/bin/env python3
"""
Measurements A + B: honest accuracy and confusion matrix, in one command.

    python validate_glove.py                 # uses ./data/*.csv, else ../04_gui/dataset.csv
    python validate_glove.py --data "data/*.csv" --fast

Data convention (no firmware or GUI changes needed): ONE CSV FILE PER SITTING,
named  <person>__<session>.csv   e.g.  will__2026-10-20_a.csv, anna__01.csv
Rows are the same  label,ch0..ch4  your 01_data_collection / GUI already writes.
In the GUI's Collect tab, hit "Browse..." and point it at a new file each time
you put the glove on. Person and session are read from the filename.

What it runs (same pipeline as the firmware minus vote/gate: per-sample argmax,
min-max constants from TRAINING data only, inputs clipped to [0,1] like nn_model.cpp):
  * random 80/20 split      -> your old number, kept for the leakage comparison
  * leave-one-session-out   -> "does it work after re-wearing the glove?"
  * leave-one-person-out    -> "does it work on someone it has never seen?"
  * confusion matrix + per-gesture precision/recall/F1 (pooled held-out folds)

If you only have ONE file (e.g. today's dataset.csv) it falls back to a
blocked-temporal CV (each hold-out is a contiguous chunk of every recording,
with a 0.5 s purge gap so neighbouring near-duplicate samples can't leak).
That is a weaker proxy and the report labels it as such.
"""
import argparse, glob, importlib.util, json, re, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAIN_PY = REPO / "02_train_model" / "train_model.py"
OUT = HERE / "results"


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_model", TRAIN_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_csv(path):
    labels, rows = [], []
    for line in open(path, newline=""):
        parts = line.strip().split(",")
        if len(parts) != 6 or parts[0].startswith("#"):
            continue
        try:
            rows.append([float(v) for v in parts[1:]])
        except ValueError:
            continue
        labels.append(parts[0].strip())
    return labels, np.array(rows, dtype=np.float32)


def parse_name(path):
    stem = Path(path).stem
    if "__" in stem:
        person, _ = stem.split("__", 1)
    else:
        person = re.split(r"[_\-]\d", stem)[0] or stem
    return person, stem


def load_data(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No files matched {pattern}")
    X, lab, person, session, runid, pos, runlen = [], [], [], [], [], [], []
    rid = 0
    for f in files:
        l, x = read_csv(f)
        if not l:
            continue
        p, s = parse_name(f)
        start = 0
        for i in range(1, len(l) + 1):  # contiguous same-label runs (for the blocked fallback)
            if i == len(l) or l[i] != l[start]:
                n = i - start
                runid += [rid] * n; pos += list(range(n)); runlen += [n] * n
                rid += 1; start = i
        X.append(x); lab += l; person += [p] * len(l); session += [s] * len(l)
    return (np.vstack(X), np.array(lab), np.array(person), np.array(session),
            np.array(runid), np.array(pos), np.array(runlen), files)


def train_mlp(tm, Xtr, ytr, n_cls, a):
    """Same MLP/Adam/batching as train_model.py, optional jitter augmentation."""
    rng = np.random.default_rng(a.seed)
    model = tm.MLP([Xtr.shape[1], *a.hidden, n_cls], seed=a.seed)
    Y = tm.one_hot(ytr, n_cls)
    for _ in range(a.epochs):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), a.batch_size):
            idx = perm[s:s + a.batch_size]
            xb = Xtr[idx]
            if a.augment > 0:  # random per-channel gain/offset, simulates strap shift
                xb = np.clip(xb * (1 + rng.normal(0, a.augment, (1, xb.shape[1])))
                             + rng.normal(0, a.augment, (1, xb.shape[1])), 0, 1).astype(np.float32)
            acts, zs = model.forward(xb)
            gW, gb = model.backward(acts, zs, Y[idx])
            model.adam_step(gW, gb, a.lr)
    return model


def predict(model, Xn):
    return np.argmax(model.forward(Xn)[0][-1], axis=1)


def fit_eval(tm, X, y, tr, te, n_cls, a, groups=None):
    if a.session_norm and groups is not None:  # simulated per-session open-hand/fist calibration
        Xs = X.copy()
        for g in np.unique(groups):
            m = groups == g
            lo, hi = X[m].min(0), X[m].max(0)
            Xs[m] = (X[m] - lo) / np.where(hi - lo < 1e-6, 1, hi - lo)
        Xtr, Xte = Xs[tr], np.clip(Xs[te], 0, 1)
    else:
        lo, hi = X[tr].min(0), X[tr].max(0)
        span = np.where(hi - lo < 1e-6, 1.0, hi - lo)
        Xtr, Xte = (X[tr] - lo) / span, np.clip((X[te] - lo) / span, 0, 1)
    model = train_mlp(tm, Xtr.astype(np.float32), y[tr], n_cls, a)
    return predict(model, Xte.astype(np.float32))


def summarize(accs, y_true, y_pred):
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "worst": float(np.min(accs)),
            "pooled": float((np.array(y_true) == np.array(y_pred)).mean()), "n_folds": len(accs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 12])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01, help="use the lr you actually deploy with")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="100 epochs, 3 random-split seeds")
    ap.add_argument("--augment", type=float, default=0.0, help="gain/offset jitter std, e.g. 0.05")
    ap.add_argument("--session-norm", action="store_true", help="simulate per-session calibration")
    ap.add_argument("--blocks", type=int, default=5, help="chunks for the single-file fallback")
    a = ap.parse_args()
    if a.fast:
        a.epochs = 100
    pattern = a.data
    if not pattern:
        d = HERE / "data"
        pattern = str(d / "*.csv") if d.exists() and list(d.glob("*.csv")) else str(REPO / "04_gui" / "dataset.csv")
    tm = load_trainer()
    X, lab, person, session, runid, pos, runlen, files = load_data(pattern)
    classes = sorted(set(lab.tolist()))
    y = np.array([classes.index(c) for c in lab])
    K = len(classes)
    sessions, persons = sorted(set(session)), sorted(set(person))
    print(f"{len(X)} samples, {K} classes {classes}, {len(sessions)} sessions, {len(persons)} people")
    t0 = time.time()
    res = {"classes": classes, "n_samples": int(len(X)), "n_sessions": len(sessions),
           "n_people": len(persons), "n_files": len(files), "per_class_samples":
           {c: int((lab == c).sum()) for c in classes},
           "settings": {"hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "augment": a.augment,
                        "session_norm": a.session_norm}}

    # random split baseline (your old number)
    seeds = range(3 if a.fast else 5)
    rs = []
    for s in seeds:
        perm = np.random.default_rng(s).permutation(len(X))
        nv = int(0.2 * len(X))
        te, tr = perm[:nv], perm[nv:]
        rs.append(float((fit_eval(tm, X, y, tr, te, K, a) == y[te]).mean()))
    res["random_split"] = {"mean": float(np.mean(rs)), "std": float(np.std(rs)), "n_seeds": len(rs)}
    print(f"random 80/20 split: {np.mean(rs):.3f} +/- {np.std(rs):.3f}")

    def logo(groups, name):
        accs, yt, yp = [], [], []
        for g in np.unique(groups):
            te = np.where(groups == g)[0]
            tr = np.where(groups != g)[0]
            if len(set(y[tr])) < K:  # held-out fold removed a whole class from training
                print(f"  skip {name} fold {g}: class missing from training"); continue
            p = fit_eval(tm, X, y, tr, te, K, a, groups=session if a.session_norm else None)
            accs.append(float((p == y[te]).mean())); yt += list(y[te]); yp += list(p)
        return accs, yt, yp

    if len(sessions) >= 3:
        res["mode"] = "leave-one-session-out"
        accs, yt, yp = logo(session, "session")
        res["held_out_sessions"] = summarize(accs, yt, yp)
        if len(persons) >= 2:
            accs_p, ytp, ypp = logo(person, "person")
            res["held_out_people"] = summarize(accs_p, ytp, ypp)
    else:
        res["mode"] = "blocked-temporal-cv (proxy; no real sessions)"
        gap = int(0.5 * 50)
        chunk = np.minimum((pos * a.blocks) // runlen, a.blocks - 1)
        accs, yt, yp = [], [], []
        for k in range(a.blocks):
            te = np.where(chunk == k)[0]
            lo_edge = np.array([pos[i] for i in te])  # purge neighbours of the held-out chunk
            tr_mask = chunk != k
            for rid in np.unique(runid[te]):
                m = runid == rid
                tp = pos[te][runid[te] == rid]
                near = m & (pos >= tp.min() - gap) & (pos <= tp.max() + gap)
                tr_mask &= ~near
            tr = np.where(tr_mask)[0]
            p = fit_eval(tm, X, y, tr, te, K, a)
            accs.append(float((p == y[te]).mean())); yt += list(y[te]); yp += list(p)
        res["held_out_sessions"] = summarize(accs, yt, yp)
    yt, yp = np.array(yt), np.array(yp)

    # B: confusion matrix + per-class scores from pooled held-out predictions
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    cm = confusion_matrix(yt, yp, labels=range(K), normalize="true")
    pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=range(K), zero_division=0)
    res["per_class"] = {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                            "support": int(sup[i])} for i, c in enumerate(classes)}
    off = [(cm[i, j], classes[i], classes[j]) for i in range(K) for j in range(K) if i != j]
    top = max(off) if off else (0, "", "")
    res["biggest_confusion"] = {"rate": float(top[0]), "true": top[1], "pred": top[2]}
    res["confusion_matrix"] = cm.tolist()
    res["runtime_s"] = round(time.time() - t0, 1)

    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / "validation.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.0 + 0.9 * K, 0.8 + 0.8 * K))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(K)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(K)); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{100*cm[i,j]:.0f}%", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_title(f"Held-out confusion matrix ({res['mode']})", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3))
        names, vals = ["random split"], [res["random_split"]["mean"]]
        names.append("held-out\nsessions"); vals.append(res["held_out_sessions"]["mean"])
        if "held_out_people" in res:
            names.append("held-out\npeople"); vals.append(res["held_out_people"]["mean"])
        ax.bar(names, [100 * v for v in vals], color=["#b0b0b0", "#3b7dd8", "#2a9d8f"][:len(vals)])
        ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 100)
        for i, v in enumerate(vals):
            ax.text(i, 100 * v + 1, f"{100*v:.1f}", ha="center", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "accuracy_comparison.png", dpi=200); plt.close(fig)
    except Exception as e:
        print("plotting skipped:", e)
    h = res["held_out_sessions"]
    print(f"{res['mode']}: {h['mean']:.3f} +/- {h['std']:.3f} (worst {h['worst']:.3f}, {h['n_folds']} folds)")
    if "held_out_people" in res:
        print(f"leave-one-person-out: {res['held_out_people']['mean']:.3f}")
    print(f"biggest confusion: {top[1]} -> {top[2]} {100*top[0]:.1f}%   -> {OUT}/")


if __name__ == "__main__":
    main()
