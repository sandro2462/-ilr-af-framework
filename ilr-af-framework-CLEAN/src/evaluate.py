"""
Evaluation: AUROC, AUPRC, sensitivity, specificity, expected calibration error (ECE),
with patient-level bootstrap 95% confidence intervals.

Usage:
    python -m src.evaluate --windows path/to/windows.npz --condition real
    python -m src.evaluate --windows path/to/windows.npz --condition augmented

The windows file is expected to contain arrays: X, y, pid, end_day, features.
This script scores an array of predicted probabilities against y; provide predictions
via --predictions (a .npy of probabilities aligned to y) or wire in your trained model.
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = prob[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def point_metrics(y_true, prob, pred=None):
    out = {
        "AUROC": float(roc_auc_score(y_true, prob)),
        "AUPRC": float(average_precision_score(y_true, prob)),
        "ECE": expected_calibration_error(y_true, prob),
    }
    if pred is not None:
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        out["sensitivity"] = tp / (tp + fn) if (tp + fn) else float("nan")
        out["specificity"] = tn / (tn + fp) if (tn + fp) else float("nan")
    return out


def patient_bootstrap_auroc(y_true, prob, pid, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    groups = {p: np.where(pid == p)[0] for p in np.unique(pid)}
    keys = np.array(list(groups.keys()))
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[p] for p in samp])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(roc_auc_score(yt, prob[idx]))
    vals = np.asarray(vals)
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--predictions", help=".npy of predicted probabilities aligned to y")
    ap.add_argument("--condition", default="real")
    args = ap.parse_args()

    z = np.load(args.windows, allow_pickle=True)
    y, pid = z["y"].astype(int), z["pid"]
    if args.predictions:
        prob = np.load(args.predictions)
    else:
        raise SystemExit("Provide --predictions, or import lopo_predict from src.train_bilstm.")

    pred = (prob >= 0.5).astype(int)
    m = point_metrics(y, prob, pred)
    med, lo, hi = patient_bootstrap_auroc(y, prob, pid)
    print(f"[{args.condition}] n={len(y)} pos={int(y.sum())} "
          f"AUROC={m['AUROC']:.3f} (boot 95% CI {lo:.2f}-{hi:.2f}) "
          f"AUPRC={m['AUPRC']:.3f} ECE={m['ECE']:.3f} "
          f"Sens={m.get('sensitivity', float('nan')):.2f} Spec={m.get('specificity', float('nan')):.2f}")


if __name__ == "__main__":
    main()
