#!/usr/bin/env python3
"""
Program 2: Train / tune the gesture classifier.

Loads the CSV(s) produced by 01_data_collection (rows: label,ch0,ch1,ch2,
ch3,ch4), trains a small feedforward network (Dense+ReLU hidden layers ->
Dense+softmax output) with plain numpy, and exports the trained weights plus
normalization constants as a C header (model_weights.h). Only numpy is
required - no TensorFlow/PyTorch, so there's nothing extra to convert before
it runs on the ESP32.

Usage:
    python train_model.py --data dataset.csv
    python train_model.py --data "data/*.csv" --hidden 24 16 --epochs 500

By default the header is written to ./model_weights.h in the folder you run
this from. Copy that file into your GestureModel library's src folder
(Documents/Arduino/libraries/GestureModel/src/model_weights.h), overwriting
the previous one, then recompile 03_inference.

Adding a new gesture:
    1. Add its name to GESTURE_LABELS in GestureShared/src/gesture_config.h
       (this is just the on-device data-collection menu).
    2. Collect data for it with 01_data_collection.
    3. Re-run this script - the output layer size and class_labels[] update
       automatically to match whatever distinct label strings are present in
       your data, in alphabetical order.
"""
import argparse
import csv
import glob
import sys

import numpy as np


def load_dataset(pattern):
    files = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not files:
        sys.exit(f"No files matched: {pattern}")

    rows, labels = [], []
    for path in files:
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                label, *features = row
                if len(features) != 5:
                    continue
                labels.append(label.strip())
                rows.append([float(x) for x in features])

    if not rows:
        sys.exit("No usable rows found in dataset.")
    return np.array(rows, dtype=np.float32), labels


def one_hot(indices, num_classes):
    out = np.zeros((len(indices), num_classes), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_true):
    eps = 1e-9
    return -np.mean(np.sum(y_true * np.log(probs + eps), axis=1))


def accuracy(probs, y_true):
    return float(np.mean(np.argmax(probs, axis=1) == np.argmax(y_true, axis=1)))


class MLP:
    """Dense+ReLU hidden layers, Dense+softmax output. Trained with Adam."""

    def __init__(self, layer_sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = layer_sizes
        self.W, self.b = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))  # Xavier/Glorot init
            self.W.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32))
            self.b.append(np.zeros(fan_out, dtype=np.float32))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        acts, zs = [X], []
        a = X
        for l in range(len(self.W)):
            z = a @ self.W[l] + self.b[l]
            zs.append(z)
            a = np.maximum(z, 0.0) if l < len(self.W) - 1 else softmax(z)
            acts.append(a)
        return acts, zs

    def backward(self, acts, zs, y_true):
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        n = y_true.shape[0]
        delta = (acts[-1] - y_true) / n  # combined softmax + cross-entropy gradient
        for l in reversed(range(len(self.W))):
            grads_W[l] = acts[l].T @ delta
            grads_b[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l].T) * (zs[l - 1] > 0)  # ReLU'
        return grads_W, grads_b

    def adam_step(self, grads_W, grads_b, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for l in range(len(self.W)):
            self.mW[l] = beta1 * self.mW[l] + (1 - beta1) * grads_W[l]
            self.vW[l] = beta2 * self.vW[l] + (1 - beta2) * grads_W[l] ** 2
            mhat = self.mW[l] / (1 - beta1 ** self.t)
            vhat = self.vW[l] / (1 - beta2 ** self.t)
            self.W[l] -= lr * mhat / (np.sqrt(vhat) + eps)

            self.mb[l] = beta1 * self.mb[l] + (1 - beta1) * grads_b[l]
            self.vb[l] = beta2 * self.vb[l] + (1 - beta2) * grads_b[l] ** 2
            mhat_b = self.mb[l] / (1 - beta1 ** self.t)
            vhat_b = self.vb[l] / (1 - beta2 ** self.t)
            self.b[l] -= lr * mhat_b / (np.sqrt(vhat_b) + eps)


def export_header(path, model, feature_min, feature_max, class_names):
    layer_sizes = model.sizes[1:]  # exclude input size
    max_width = max(model.sizes)

    def fmt_array(arr):
        return ", ".join(f"{x:.8f}f" for x in arr.flatten())

    lines = [
        "// AUTO-GENERATED by 02_train_model/train_model.py - do not hand-edit.",
        "#pragma once",
        "",
        f"#define NN_INPUT_SIZE {model.sizes[0]}",
        f"#define NN_NUM_LAYERS {len(layer_sizes)}",
        f"#define NN_MAX_WIDTH {max_width}",
        f"#define NUM_GESTURES {len(class_names)}",
        "",
        f"static const int nn_layer_sizes[NN_NUM_LAYERS] = "
        f"{{{', '.join(str(s) for s in layer_sizes)}}};",
        "",
    ]

    for i, (W, b) in enumerate(zip(model.W, model.b)):
        # C side expects row-major [out_size x in_size]; numpy W is [in_size x out_size].
        Wt = W.T
        lines.append(f"static const float nn_weights_{i}[{Wt.size}] = {{{fmt_array(Wt)}}};")
        lines.append(f"static const float nn_biases_{i}[{b.size}] = {{{fmt_array(b)}}};")
        lines.append("")

    w_ptrs = ", ".join(f"nn_weights_{i}" for i in range(len(model.W)))
    b_ptrs = ", ".join(f"nn_biases_{i}" for i in range(len(model.b)))
    lines.append(f"static const float* const nn_weights[NN_NUM_LAYERS] = {{{w_ptrs}}};")
    lines.append(f"static const float* const nn_biases[NN_NUM_LAYERS] = {{{b_ptrs}}};")
    lines.append("")

    labels_c = ", ".join(f'"{n}"' for n in class_names)
    lines.append(f"static const char* const class_labels[NUM_GESTURES] = {{{labels_c}}};")
    lines.append("")
    lines.append(f"static const float feature_min[NN_INPUT_SIZE] = {{{fmt_array(feature_min)}}};")
    lines.append(f"static const float feature_max[NN_INPUT_SIZE] = {{{fmt_array(feature_max)}}};")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV file or glob pattern, e.g. data/*.csv")
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 12], help="hidden layer sizes")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--out", default="model_weights.h",
                     help="where to write the generated header (default: model_weights.h "
                          "in the current folder - copy it into your GestureModel library's "
                          "src folder afterward, see README)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, label_strings = load_dataset(args.data)
    class_names = sorted(set(label_strings))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([class_to_idx[s] for s in label_strings])

    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)
    span = np.where(feature_max - feature_min < 1e-6, 1.0, feature_max - feature_min)
    Xn = (X - feature_min) / span

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xn))
    Xn, y_idx = Xn[perm], y_idx[perm]
    n_val = int(len(Xn) * args.val_split)
    X_val, y_val = Xn[:n_val], y_idx[:n_val]
    X_tr, y_tr = Xn[n_val:], y_idx[n_val:]

    Y_tr = one_hot(y_tr, len(class_names))
    Y_val = one_hot(y_val, len(class_names))

    layer_sizes = [X.shape[1], *args.hidden, len(class_names)]
    model = MLP(layer_sizes, seed=args.seed)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train/val sizes: {len(X_tr)}/{len(X_val)}  Architecture: {layer_sizes}")

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(len(X_tr))
        for start in range(0, len(X_tr), args.batch_size):
            idx = perm[start:start + args.batch_size]
            acts, zs = model.forward(X_tr[idx])
            gW, gb = model.backward(acts, zs, Y_tr[idx])
            model.adam_step(gW, gb, args.lr)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            tr_probs, _ = model.forward(X_tr)
            val_acc = float("nan")
            if n_val > 0:
                val_probs, _ = model.forward(X_val)
                val_acc = accuracy(val_probs[-1], Y_val)
            tr_acc = accuracy(tr_probs[-1], Y_tr)
            loss = cross_entropy(tr_probs[-1], Y_tr)
            print(f"epoch {epoch:4d}  loss {loss:.4f}  train_acc {tr_acc:.3f}  val_acc {val_acc:.3f}")

    export_header(args.out, model, feature_min, feature_max, class_names)


if __name__ == "__main__":
    main()
