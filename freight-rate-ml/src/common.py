"""Shared cleaning + feature engineering for the freight-rate model."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_EQUIP = ["Dry Van", "Flatbed", "Reefer"]


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """market_index is a daily market level plus ~0.025 row noise -> daily median is a clean estimate.
    Uses feature columns only (no labels), so pooling train + validation rows is leakage-free."""
    allf = pd.concat([f[["date", "market_index"]] for f in frames])
    return allf.groupby("date")["market_index"].median()


def daily_quote_signal(*frames: pd.DataFrame) -> pd.Series:
    allf = pd.concat([f[["date", "quote_signal"]] for f in frames])
    return allf.groupby("date")["quote_signal"].median()


def weight_medians(train: pd.DataFrame) -> dict:
    return train.assign(weight=train["weight"].abs()).groupby("equipment")["weight"].median().to_dict()


def clean_features(df: pd.DataFrame, w_med: dict, daily_mi: pd.Series) -> pd.DataFrame:
    """Row-level repairs. Never touches the label."""
    out = df.copy()
    # 1) negative weights are sign flips (|min| = 5,000 lb is the physical floor) -> abs()
    out["weight_was_neg"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    # 2) missing weight -> equipment median (+ indicator)
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].fillna(out["equipment"].map(w_med))
    # 3) missing market_index -> that day's market level
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["date"].map(daily_mi))
    return out


def add_features(df: pd.DataFrame, daily_mi: pd.Series, use_market: bool = True,
                 use_signal: bool = True, use_calendar: bool = False) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["log_distance"] = np.log(df["distance"])
    x["weight_k"] = df["weight"] / 1000.0
    x["equip"] = pd.Categorical(df["equipment"], categories=CAT_EQUIP).codes
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        x[c] = df[c]
    x["dlat"] = df["delivery_lat"] - df["pickup_lat"]
    x["dlon"] = df["delivery_lon"] - df["pickup_lon"]
    x["dow"] = df["date"].dt.dayofweek
    if use_market:
        x["market_index"] = df["market_index"]
        # market level relative to its trailing 30-day average (momentum)
        mi = daily_mi.sort_index()
        trail = mi.rolling(30, min_periods=5).mean().shift(1)
        x["mi_vs_30d"] = (df["date"].map(mi) / df["date"].map(trail)).fillna(1.0)
    if use_signal:
        x["quote_signal"] = df["quote_signal"]
    if use_calendar:
        x["doy"] = df["date"].dt.dayofyear
    return x
