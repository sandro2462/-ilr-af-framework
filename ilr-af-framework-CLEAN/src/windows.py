"""
Window construction: 30-day windows, stride 1, with a 7-day AF outcome.

The primary outcome for a window ending on day t is any device-detected AF episode
(daily AT/AF burden > 0) within days t+1 .. t+7. Stride-1 windows maximise temporal
resolution but introduce autocorrelation between consecutive samples; this is handled
downstream by patient-level (leave-one-patient-out) cross-validation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WINDOW = 30
HORIZON = 7


def make_windows(df: pd.DataFrame, features: list[str],
                 window: int = WINDOW, horizon: int = HORIZON):
    """Return X (n, window, n_features), y (n,), pid (n,), end_day (n,)."""
    X, y, pid, end_day = [], [], [], []
    for p, g in df.sort_values(["Patient_ID", "Day_Index"]).groupby("Patient_ID"):
        vals = g[features].to_numpy(dtype="float32")
        burden = (g["AF_burden_pct"].to_numpy() > 0).astype(int)
        n = len(g)
        for t in range(window - 1, n - horizon):
            X.append(vals[t - window + 1: t + 1])
            y.append(int(burden[t + 1: t + 1 + horizon].any()))
            pid.append(p)
            end_day.append(int(g["Day_Index"].iloc[t]))
    return (np.asarray(X, dtype="float32"),
            np.asarray(y, dtype="int8"),
            np.asarray(pid),
            np.asarray(end_day, dtype="int64"))
