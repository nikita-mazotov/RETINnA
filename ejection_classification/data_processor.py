# data_processor.py
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

REQUIRED = ["mass", "rTarget", "x0", "y0", "vx0", "vy0", "ejected"]

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Require the new standard columns
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce numerics
    for c in ["mass", "rTarget", "x0", "y0", "vx0", "vy0"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize label to {0,1}
    lab = df["ejected"].astype(str).str.strip().str.lower()
    mapping = {"true":1,"t":1,"1":1,"yes":1,"y":1,"false":0,"f":0,"0":0,"no":0,"n":0}
    df["ejected"] = lab.map(mapping).astype(float)

    # Clean NaNs/Infs in inputs + label
    keep_cols = REQUIRED[:]  # copy
    if "earth_af" in df.columns:
        # keep column for downstream analysis; not a feature
        df["earth_af"] = pd.to_numeric(df["earth_af"], errors="coerce")
        keep_cols.append("earth_af")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols).copy()
    df["ejected"] = df["ejected"].astype(int)

    # Drop non-feature identifiers if present
    for junk in ("index",):
        if junk in df.columns:
            df.drop(columns=[junk], inplace=True)

    return df

def engineer_features(df: pd.DataFrame, *, eps: float = 1e-12) -> pd.DataFrame:
    """
    Polar-only, rotation-aware features + scalar params:
      r0, v0, cos_dtheta, sin_dtheta, mass, rTarget
    """
    x  = df["x0"].to_numpy()
    y  = df["y0"].to_numpy()
    vx = df["vx0"].to_numpy()
    vy = df["vy0"].to_numpy()

    r0 = np.sqrt(x*x + y*y)
    v0 = np.sqrt(vx*vx + vy*vy)
    theta_r = np.arctan2(y,  x)
    theta_v = np.arctan2(vy, vx)
    dtheta = (theta_v - theta_r + np.pi) % (2*np.pi) - np.pi

    feats = pd.DataFrame({
        "r0": r0,
        "v0": np.where(v0 < eps, eps, v0),
        "cos_dtheta": np.cos(dtheta),
        "sin_dtheta": np.sin(dtheta),
        "mass": df["mass"].to_numpy(),
        "rTarget": df["rTarget"].to_numpy(),
    }, index=df.index)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.dropna(how="any", inplace=True)
    return feats

def _safe_stratify(y: pd.Series, enable: bool):
    if not enable:
        return None
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
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=_safe_stratify(y, stratify)
    )
    val_adj = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_adj, random_state=random_state,
        stratify=_safe_stratify(y_tmp, stratify)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_features_and_split3(
    csv_path: str | Path,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    stratify: bool = True,
):
    df = load_csv(csv_path)
    X = engineer_features(df)
    y = df.loc[X.index, "ejected"].astype(int)  # align after any row drops
    return train_val_test_split_stratified(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )

def class_weight_hint(y_train: pd.Series) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return float(neg / max(pos, 1))
