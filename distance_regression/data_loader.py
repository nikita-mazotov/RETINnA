# data_regression.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Expected columns in the new regression dataset
REQUIRED = ["mass", "rTarget", "x0", "y0", "vx0", "vy0", "earth_af"]

def load_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV, ensure required columns, coerce numerics, drop NaNs/infs."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Coerce numerics
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED).copy()

    # Target is semimajor axis after 100 yrs (can be negative for unbound)
    df["earth_af"] = df["earth_af"].astype(float)
    return df

def engineer_features_polar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Polar-only features (no Cartesian duplication), matching prior convention.
      - r0, v0
      - cos_dtheta, sin_dtheta  where dtheta = atan2(vy, vx) - atan2(y, x) wrapped to [-pi, pi]
      - mass, rTarget  (as-provided)
    """
    x, y, vx, vy = (df["x0"].values, df["y0"].values, df["vx0"].values, df["vy0"].values)
    r0 = np.sqrt(x*x + y*y)
    v0 = np.sqrt(vx*vx + vy*vy)

    theta_r = np.arctan2(y, x)
    theta_v = np.arctan2(vy, vx)
    dtheta  = (theta_v - theta_r + np.pi) % (2.0*np.pi) - np.pi

    feats = pd.DataFrame({
        "r0": r0,
        "v0": v0,
        "cos_dtheta": np.cos(dtheta),
        "sin_dtheta": np.sin(dtheta),
        "mass": df["mass"].values.astype(float),
        "rTarget": df["rTarget"].values.astype(float),
    }, index=df.index)

    # Optional: sanity info if any zero radii (avoid divide-by-zero elsewhere)
    # (We don't compute vr/vt here to keep strictly to polar-angular features.)
    return feats

def _stratify_bins_for_regression(y: pd.Series, n_bins: int, min_count: int = 2):
    """
    Create quantile bins for y to allow stratified splitting in regression.
    Returns an array of bin labels or None if bins would be too sparse.
    """
    if not n_bins or n_bins < 2:
        return None
    yv = np.asarray(y, dtype=float)
    # Guard: if all targets identical, can't bin
    if np.nanmax(yv) == np.nanmin(yv):
        return None
    # Quantile edges; unique to avoid duplicates from ties
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.nanquantile(yv, qs))
    if edges.size < 3:  # need at least 2 bins
        return None
    bins = np.digitize(yv, edges[1:-1], right=True)
    # Ensure every bin has at least min_count
    counts = np.bincount(bins, minlength=len(edges)-1)
    if (counts < min_count).any():
        return None
    return bins

def train_val_test_split3(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    stratify_bins: int = 0,   # e.g. 5 or 10 to enable quantile stratification
):
    """
    Two-stage split: (train+val) vs test, then train vs val.
    Optionally stratify by quantile bins of y (regression-friendly).
    """
    # First split: hold-out test
    strat_all = _stratify_bins_for_regression(y, stratify_bins)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_all
    )

    # Second split: train vs val (bins recomputed on temp to avoid leakage)
    val_size_adj = val_size / (1.0 - test_size)
    strat_temp = _stratify_bins_for_regression(y_temp, stratify_bins)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adj,
        random_state=random_state,
        stratify=strat_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_features_and_split3(
    csv_path: str | Path,
    *,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    stratify_bins: int = 0,
):
    """
    Convenience wrapper: read CSV -> engineer polar features -> 3-way split.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = load_csv(csv_path)
    X = engineer_features_polar(df)
    y = df["earth_af"].astype(float)
    return train_val_test_split3(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify_bins=stratify_bins,
    )
