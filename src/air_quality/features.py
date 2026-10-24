"""Feature engineering and causal transformation pipelines.

This module provides causal, leakage-free engineered features for air quality
sensor calibration, adhering strictly to ``docs/data_contract.md`` §10.

Features are computed on the complete hourly grid returned by
:func:`air_quality.data.load_raw` before any row filtering. All windows look
strictly backwards, with a maximum look-back window of 24 hours.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from air_quality.data import (
    ENV_COLS,
    FORBIDDEN_COLS,
    INPUT_COLS,
    SENSOR_COLS,
    TARGET,
    inputs_available,
)

# Supported feature groups
FEATURE_GROUPS = ("time", "lags", "rolling", "env_interaction", "missing_ind")
REQUIRED_GROUPS = ("time", "lags")
ALL_GROUPS = FEATURE_GROUPS

# Longest look-back window across all engineered features (in hours)
LONGEST_LOOKBACK_HOURS = 24


def _get_hourly_grid(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a complete hourly DatetimeIndex spanning df.index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex aligned to the hour.")
    return pd.date_range(df.index.min(), df.index.max(), freq="h", name=df.index.name)


def _build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct cyclical and calendar temporal features from timestamp index.

    Features:
    - ``hour_sin``, ``hour_cos``: 24-hour diurnal cyclical encoding.
    - ``dow_sin``, ``dow_cos``: 7-day weekly cyclical encoding.
    - ``dayofweek``: Integer day of the week (Monday=0, Sunday=6).
    - ``is_weekend``: Binary flag for Saturday and Sunday.
    """
    ts = df.index
    hour = ts.hour
    dow = ts.dayofweek

    feats = pd.DataFrame(index=ts)
    feats["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    feats["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    feats["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    feats["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    feats["dayofweek"] = dow.astype(float)
    feats["is_weekend"] = (dow >= 5).astype(float)
    return feats


def _build_lag_features(
    df: pd.DataFrame,
    lag_hours: Sequence[int] = (1, 3, 24),
    cols: Sequence[str] = SENSOR_COLS,
) -> pd.DataFrame:
    """Construct backwards-looking lags for low-cost sensors on the hourly grid.

    Lags are computed on a complete hourly grid to ensure that shift(k) represents
    exactly k hours. A lag across an unobserved gap produces NaN, never bridging
    missing periods silently.
    """
    grid = _get_hourly_grid(df)
    lag_cols: dict[str, pd.Series] = {}

    for k in lag_hours:
        if k <= 0:
            raise ValueError(f"Lag hours must be strictly positive (backward-looking), got {k}")
        for col in cols:
            if col not in df.columns:
                raise KeyError(f"Required input column '{col}' missing from DataFrame.")
            # Reindex to full hourly grid before shifting to respect time intervals
            s_grid = df[col].reindex(grid)
            lagged = s_grid.shift(k).reindex(df.index)
            lag_cols[f"{col}_lag_{k}h"] = lagged

    return pd.DataFrame(lag_cols, index=df.index)


def _build_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (6, 24),
    cols: Sequence[str] = ("PT08.S1(CO)",),
) -> pd.DataFrame:
    """Construct past-only rolling summary statistics.

    Crucial rule: shift(1) is applied BEFORE rolling, ensuring that the current
    hour's observation at time t is strictly excluded from past rolling summaries.
    """
    grid = _get_hourly_grid(df)
    roll_cols: dict[str, pd.Series] = {}

    for w in windows:
        if w <= 1:
            raise ValueError(f"Rolling window must be > 1 hour, got {w}")
        for col in cols:
            if col not in df.columns:
                raise KeyError(f"Required input column '{col}' missing from DataFrame.")
            s_grid = df[col].reindex(grid)
            # Past-only: shift(1) excludes time t
            past_only = s_grid.shift(1)
            mean_series = past_only.rolling(window=w, min_periods=1).mean().reindex(df.index)
            std_series = past_only.rolling(window=w, min_periods=2).std().reindex(df.index)

            roll_cols[f"{col}_roll_mean_{w}h"] = mean_series
            roll_cols[f"{col}_roll_std_{w}h"] = std_series

    return pd.DataFrame(roll_cols, index=df.index)


def _build_env_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct thermodynamic and environmental interaction terms.

    Captures known sensor conductivity sensitivity to temperature and humidity:
    - ``T_x_RH``: Temperature × Relative Humidity
    - ``PT08.S1_x_T``: Tin-oxide CO sensor × Temperature
    - ``PT08.S1_x_AH``: Tin-oxide CO sensor × Absolute Humidity
    """
    feats = pd.DataFrame(index=df.index)
    if "T" in df.columns and "RH" in df.columns:
        feats["T_x_RH"] = df["T"] * df["RH"]
    if "PT08.S1(CO)" in df.columns and "T" in df.columns:
        feats["PT08.S1_x_T"] = df["PT08.S1(CO)"] * df["T"]
    if "PT08.S1(CO)" in df.columns and "AH" in df.columns:
        feats["PT08.S1_x_AH"] = df["PT08.S1(CO)"] * df["AH"]
    return feats


def _build_missing_indicator_features(
    df: pd.DataFrame,
    lag_hours: Sequence[int] = (1, 24),
) -> pd.DataFrame:
    """Construct past binary missingness indicators for low-cost sensor outages.

    Indicates whether sensor inputs were unobserved at past intervals t - k.
    Looks strictly backwards without imputing artificial values.
    """
    grid = _get_hourly_grid(df)
    missing_series = (~inputs_available(df)).astype(float).reindex(grid)
    miss_cols: dict[str, pd.Series] = {}

    for k in lag_hours:
        miss_cols[f"inputs_missing_lag_{k}h"] = missing_series.shift(k).reindex(df.index)

    return pd.DataFrame(miss_cols, index=df.index)


def build_features(
    df: pd.DataFrame,
    groups: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Build a timestamp-indexed DataFrame of engineered features from named groups.

    Parameters
    ----------
    df : pd.DataFrame
        Complete hourly grid DataFrame (typically as returned by
        :func:`air_quality.data.load_raw`). Must have a DatetimeIndex aligned
        to the hour.
    groups : sequence of str, str, or None
        Names of feature groups to construct. Supported values:
        - ``"time"``: hour_sin, hour_cos, dow_sin, dow_cos, dayofweek, is_weekend
        - ``"lags"``: low-cost sensor lags at 1h, 3h, 24h
        - ``"rolling"``: past-only 6h and 24h rolling mean and std of PT08.S1(CO)
        - ``"env_interaction"``: T*RH, PT08.S1*T, PT08.S1*AH
        - ``"missing_ind"``: binary outage indicators at past lags (1h, 24h)
        - ``"all"``: builds all available groups.
        If None, defaults to ``REQUIRED_GROUPS`` (``"time"`` and ``"lags"``).

    Returns
    -------
    pd.DataFrame
        Timestamp-indexed DataFrame containing the selected engineered features.
    """
    if groups is None:
        selected_groups = list(REQUIRED_GROUPS)
    elif isinstance(groups, str):
        if groups == "all":
            selected_groups = list(ALL_GROUPS)
        else:
            selected_groups = [groups]
    else:
        selected_groups = list(groups)

    # Validate group names
    unknown = [g for g in selected_groups if g not in ALL_GROUPS and g != "all"]
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}. Available groups: {ALL_GROUPS}")

    if "all" in selected_groups:
        selected_groups = list(ALL_GROUPS)

    parts: list[pd.DataFrame] = []

    if "time" in selected_groups:
        parts.append(_build_time_features(df))

    if "lags" in selected_groups:
        parts.append(_build_lag_features(df))

    if "rolling" in selected_groups:
        parts.append(_build_rolling_features(df))

    if "env_interaction" in selected_groups:
        parts.append(_build_env_interaction_features(df))

    if "missing_ind" in selected_groups:
        parts.append(_build_missing_indicator_features(df))

    if not parts:
        return pd.DataFrame(index=df.index)

    return pd.concat(parts, axis=1)
