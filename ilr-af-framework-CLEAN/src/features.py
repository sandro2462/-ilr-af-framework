"""
Feature engineering: raw daily ILR parameters -> 18 features.

Daily device-reported parameters:
    night heart rate, day heart rate, HRV, activity (ADL minutes), AT/AF burden.

Features are computed per patient and then z-scored within patient. They are partitioned
into 16 autonomic/activity features and 2 AF-derived features so that anticipatory
contributions can be assessed independently of ongoing arrhythmia.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

AUTONOMIC_FEATURES = [
    "HRR", "dHR", "HRV_d3", "HR_d3", "Act_d3",
    "HRV_roll7", "HRV_std7", "HR_roll7", "HRn_roll7",
    "Act_roll7", "Act_std7", "dHR_roll7",
    "NightHeartRate_bpm_z", "DayHeartRate_bpm_z",
    "HeartRateVariability_ms_z", "ActivitiesOfDlyLiving_minutes_z",
]  # 16 autonomic / activity features

AF_FEATURES = ["AF_burden_pct", "AF_slope"]  # 2 AF-derived features

ALL_FEATURES = AUTONOMIC_FEATURES + AF_FEATURES  # 18 features


def _zscore_within_patient(df: pd.DataFrame, cols: list[str], group: str = "Patient_ID") -> pd.DataFrame:
    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0
    out = df.copy()
    for c in cols:
        out[c] = df.groupby(group)[c].transform(z)
    return out


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute the 18 features from a daily dataframe (one row per patient-day).

    Expected input columns: Patient_ID, NightHeartRate_bpm, DayHeartRate_bpm,
    HeartRateVariability_ms, ActivitiesOfDlyLiving_minutes, TimeInATAF_ms.
    """
    df = daily.sort_values(["Patient_ID", "Day_Index"]).copy()
    g = df.groupby("Patient_ID", group_keys=False)

    # autonomic / activity
    df["HRR"] = df["NightHeartRate_bpm"] / df["DayHeartRate_bpm"]
    df["dHR"] = df["DayHeartRate_bpm"] - df["NightHeartRate_bpm"]
    df["HRV_d3"] = g["HeartRateVariability_ms"].apply(lambda s: s.diff(3))
    df["HR_d3"] = g["DayHeartRate_bpm"].apply(lambda s: s.diff(3))
    df["Act_d3"] = g["ActivitiesOfDlyLiving_minutes"].apply(lambda s: s.diff(3))
    for col, name in [("HeartRateVariability_ms", "HRV"), ("DayHeartRate_bpm", "HR"),
                      ("NightHeartRate_bpm", "HRn"), ("ActivitiesOfDlyLiving_minutes", "Act")]:
        df[f"{name}_roll7"] = g[col].apply(lambda s: s.rolling(7, min_periods=1).mean())
    for col, name in [("HeartRateVariability_ms", "HRV"), ("ActivitiesOfDlyLiving_minutes", "Act")]:
        df[f"{name}_std7"] = g[col].apply(lambda s: s.rolling(7, min_periods=1).std())
    df["dHR_roll7"] = g.apply(lambda x: x["dHR"].rolling(7, min_periods=1).mean()).reset_index(drop=True)

    # AF-derived
    df["AF_burden_pct"] = df["TimeInATAF_ms"] / (24 * 60 * 60 * 1000.0) * 100.0
    df["AF_slope"] = g["AF_burden_pct"].apply(lambda s: s.diff(3))

    # z-scores of raw signals used directly as features
    raw_z = ["NightHeartRate_bpm", "DayHeartRate_bpm",
             "HeartRateVariability_ms", "ActivitiesOfDlyLiving_minutes"]
    df = _zscore_within_patient(df, raw_z)
    df = df.rename(columns={c: f"{c}_z" for c in raw_z})

    return df
