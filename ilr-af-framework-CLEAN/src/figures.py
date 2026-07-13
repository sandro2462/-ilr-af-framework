"""
Generate the summary table (Table 1) and figures from evaluation outputs.

Table 1 rows: BiLSTM augmented (all 18), AT/AF burden alone, BiLSTM augmented (autonomic),
BiLSTM real (all 18). Provide a dict of metric rows and this writes a CSV/markdown table.
"""
from __future__ import annotations
import csv


def write_table1(rows: list[dict], path: str = "table1.csv"):
    """rows: list of dicts with keys
    analysis, features, auroc_ci, sens, spec, auprc."""
    cols = ["analysis", "features", "auroc_ci", "sens", "spec", "auprc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


if __name__ == "__main__":
    example = [
        {"analysis": "BiLSTM, augmented", "features": "18 (all)",
         "auroc_ci": "0.84 (0.73-0.93)", "sens": "68%", "spec": "98%", "auprc": "0.10"},
        {"analysis": "AT/AF burden alone", "features": "1 (AT/AF burden)",
         "auroc_ci": "0.86 (0.74-0.95)", "sens": "50%", "spec": "97%", "auprc": "0.03"},
        {"analysis": "BiLSTM, augmented", "features": "16 (autonomic)",
         "auroc_ci": "0.89 (0.83-0.94)", "sens": "62%", "spec": "93%", "auprc": "0.05"},
        {"analysis": "BiLSTM, real data", "features": "18 (all)",
         "auroc_ci": "0.57 (0.46-0.68)", "sens": "34%", "spec": "72%", "auprc": "0.03"},
    ]
    print("wrote", write_table1(example))
