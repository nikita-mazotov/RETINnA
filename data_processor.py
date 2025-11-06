# data_processor.py 
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

REQUIRED = ["x","y","vx","vy","ejected"]

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Normalize label to {0,1}
    lab = df["ejected"].astype(str).str.strip().str.lower()
    mapping = {"true":1,"t":1,"1":1,"yes":1,"y":1,"false":0,"f":0,"0":0,"no":0,"n":0}
    df["ejected"] = lab.map(mapping)

    # Coerce numerics & clean
    for c in ["x","y","vx","vy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["x","y","vx","vy","ejected"]).copy()
    df["ejected"] = df["ejected"].astype(int)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    x,y,vx,vy = df["x"], df["y"], df["vx"], df["vy"]
    r0 = np.sqrt(x**2 + y**2)
    v0 = np.sqrt(vx**2 + vy**2)
    vr = (x*vx + y*vy) / (r0 + eps)
    vt = (x*vy - y*vx) / (r0 + eps)
    Lz = x*vy - y*vx
    theta_r = np.arctan2(y, x)
    theta_v = np.arctan2(vy, vx)
    dtheta = (theta_v - theta_r + np.pi) % (2*np.pi) - np.pi
    return pd.DataFrame({
        "x":x, "y":y, "vx":vx, "vy":vy,
        "r0":r0, "v0":v0, "vr":vr, "vt":vt, "Lz":Lz, "dtheta":dtheta
    }, index=df.index)

def _safe_stratify(y: pd.Series, enable: bool):
    """Return y if stratification is feasible, else None (fallback to non-stratified)."""
    if not enable:
        return None
    # Each class needs at least 2 samples for any split; we’ll rely on try/except at call sites.
    vc = y.value_counts()
    if (vc < 2).any():
        return None
    return y

def train_val_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    stratify: bool = True,
):
    """
    Two-stage split with stratification:
      1) temp/test split with test_size
      2) train/val split on temp so that overall val fraction ~= val_size
    """
    # First split -> hold-out test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=_safe_stratify(y, stratify)
    )

    # Compute adjusted validation fraction on the remaining temp set
    val_size_adj = val_size / (1.0 - test_size)

    # Second split -> train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adj,
        random_state=random_state,
        stratify=_safe_stratify(y_temp, stratify)
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_features_and_split3(
    csv_path: str | Path,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    stratify: bool = True
):
    """
    Convenience: read CSV, build features, return 3-way split.
    """
    df = load_csv(csv_path)
    X = engineer_features(df)
    y = df["ejected"].astype(int)

    return train_val_test_split_stratified(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )

def class_weight_hint(y_train: pd.Series) -> float:
    """
    For imbalanced classes, return n_negative / n_positive,
    useful for XGBoost's `scale_pos_weight` or BCE class weighting.
    """
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    return float(neg / max(pos, 1))
