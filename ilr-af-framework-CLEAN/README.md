# A Deep-Learning Framework for Learning Candidate Pre-Arrhythmic Physiological Signatures from Continuous ILR Monitoring

Reference implementation of the analysis pipeline for the Short Report submitted to
*European Heart Journal – Digital Health*.

This repository contains the **code** that implements the framework described in the paper:
feature engineering, 30-day windowing, a bidirectional LSTM trained with
leave-one-patient-out (LOPO) cross-validation, clinician-supervised synthetic augmentation
(TIMA / TimeGAN), the feature ablation, and the evaluation protocol.

> **It does not contain patient-level data.** The underlying implantable loop recorder (ILR)
> records are personal health data held under a data use agreement and are available under
> controlled access only (see `data/README.md`). Only synthetic data and code are distributed here.

---

## What this repository provides

- A faithful, runnable reference implementation of the pipeline reported in the manuscript.
- The evaluation code (AUROC, AUPRC, expected calibration error, patient-level bootstrap CIs).
- The synthetic-data generation interface (TIMA / TimeGAN).
- Scripts to reproduce the tables and figures **from an appropriately licensed data file**.

## Methodology (summary)

- **Input:** daily ILR-reported parameters (night/day heart rate, HRV, activity, AT/AF burden).
- **Features:** 18 features derived by within-patient z-score normalisation, partitioned into
  16 autonomic/activity features and 2 AF-derived features (AT/AF burden, AF-burden slope).
- **Windows:** 30-day windows, stride 1; outcome = any device-detected AF (AT/AF burden > 0)
  within the following 7 days.
- **Model:** bidirectional LSTM, LOPO cross-validation.
- **Augmentation:** class-conditional TimeGAN under the clinician-supervised TIMA protocol.
- **Ablation:** (A) all 18 features; (B) 16 autonomic-only; (C) AT/AF burden alone.
- **Metrics:** AUROC, AUPRC, sensitivity, specificity, expected calibration error, with
  patient-level bootstrap confidence intervals.

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── src/
│   ├── features.py        # raw daily parameters -> 18 engineered features
│   ├── windows.py         # 30-day windowing (stride 1) + 7-day outcome
│   ├── train_bilstm.py    # BiLSTM + leave-one-patient-out cross-validation
│   ├── augment_tima.py    # TimeGAN / TIMA synthetic augmentation interface
│   ├── ablation.py        # feature-domain ablation (A / B / C)
│   ├── evaluate.py        # AUROC / AUPRC / ECE + patient-level bootstrap
│   └── figures.py         # table and figure generation
└── data/
    ├── README.md          # data availability and controlled-access statement
    └── synthetic/         # TIMA-generated synthetic data (non-identifiable)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the analysis

Patient-level data are under controlled access and are **not** included. With an
appropriately licensed windowed data file (see `data/README.md`), the evaluation can be
reproduced as follows:

```bash
python -m src.evaluate --windows path/to/windows.npz --condition real
python -m src.evaluate --windows path/to/windows.npz --condition augmented
```

## Data availability

The underlying ILR data were collected under a data use agreement and are available from the
corresponding author upon reasonable request, subject to institutional data-sharing agreements
and applicable privacy regulations. See `data/README.md`. Synthetic data generated under the
TIMA protocol are provided in `data/synthetic/` and are non-identifiable by construction.

## Citation

If you use this code, please cite the paper and this repository.

10.5281/zenodo.21339045
Parise G, Ceravolo R, Lucà F, De Asmundis C, La Meir M, Gulizia M, Parise O, Gelsomino S.
A deep-learning framework for learning candidate pre-arrhythmic physiological signatures
from continuous ILR monitoring. European Heart Journal – Digital Health (under review).

Software archive: 10.5281/zenodo.21339045
```

## License

Code is released under the MIT License (see `LICENSE`).
