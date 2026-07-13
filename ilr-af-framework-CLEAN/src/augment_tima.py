"""
Synthetic augmentation under the clinician-supervised TIMA protocol.

Class-conditional TimeGAN is used to expand the minority (positive) class in the TRAINING
fold only. TIMA adds structured multidisciplinary oversight over data synthesis: constraint
definition, physiological-plausibility assessment, and iterative refinement. Synthetic quality
is checked with total variation distance, Kolmogorov-Smirnov statistics, PCA variance ratios,
and nearest-neighbour diversity.

This module defines the interface; plug in the TimeGAN implementation used in your environment.
"""
from __future__ import annotations
import numpy as np


def timegan_generate(X_pos: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Return n synthetic positive windows shaped like X_pos[:, :, :].

    Replace the body with a call to your TimeGAN implementation (e.g. ydata-synthetic).
    """
    raise NotImplementedError("Wire in the TimeGAN implementation used for the study.")


def tima_augment(Xtr: np.ndarray, ytr: np.ndarray, ratio: float = 1.0, seed: int = 0):
    """Augment the training fold by generating synthetic positives up to `ratio`.

    Only the training fold is augmented; evaluation is always on real held-out windows.
    """
    pos = Xtr[ytr == 1]
    n_target = int(ratio * (ytr == 0).sum()) - (ytr == 1).sum()
    if n_target <= 0 or len(pos) == 0:
        return Xtr, ytr
    synth = timegan_generate(pos, n_target, seed=seed)
    X_aug = np.concatenate([Xtr, synth], axis=0)
    y_aug = np.concatenate([ytr, np.ones(len(synth), dtype=ytr.dtype)])
    return X_aug, y_aug
