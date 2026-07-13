"""
Feature-domain ablation to disentangle anticipation from detection.

Configurations:
    A: all 18 features
    B: 16 autonomic-only features (exclude AT/AF burden and AF-burden slope)
    C: AT/AF burden alone (single predictor)
"""
from __future__ import annotations
import numpy as np
from .features import ALL_FEATURES, AUTONOMIC_FEATURES, AF_FEATURES


def select(X: np.ndarray, feature_names: list[str], config: str) -> tuple[np.ndarray, list[str]]:
    idx = {name: i for i, name in enumerate(feature_names)}
    if config == "A":
        cols = [idx[f] for f in ALL_FEATURES if f in idx]
    elif config == "B":
        cols = [idx[f] for f in AUTONOMIC_FEATURES if f in idx]
    elif config == "C":
        cols = [idx[f] for f in ["AF_burden_pct"] if f in idx]
    else:
        raise ValueError("config must be 'A', 'B' or 'C'")
    return X[:, :, cols], [feature_names[c] for c in cols]
