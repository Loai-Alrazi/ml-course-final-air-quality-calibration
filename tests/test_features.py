"""Tests for feature engineering, causal temporal isolation, and leakage prevention.

Implements the acceptance criteria for Issue #3 (docs/data_contract.md §10):
1. Causal backward-looking verification: modifying t + k does not alter features at t.
2. Deployment & target leakage prevention: FORBIDDEN_COLS and CO(GT) are never inputs.
3. Hourly grid gap integrity: lags across unobserved hours produce NaN, not previous rows.
4. Interface flexibility: group selection and validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from air_quality.data import (
    FORBIDDEN_COLS,
    INPUT_COLS,
    SENSOR_COLS,
    TARGET,
)
from air_quality.features import (
    LONGEST_LOOKBACK_HOURS,
    build_features,
)


@pytest.fixture
def synthetic_hourly_grid() -> pd.DataFrame:
    """Create a synthetic 120-hour complete hourly grid with simulated outage blocks."""
    start = pd.Timestamp("2004-04-01 00:00:00")
    index = pd.date_range(start, periods=120, freq="h", name="timestamp")
    rng = np.random.default_rng(42)

    data = {col: rng.normal(1000.0, 100.0, size=len(index)) for col in SENSOR_COLS}
    data["T"] = rng.uniform(10.0, 30.0, size=len(index))
    data["RH"] = rng.uniform(20.0, 80.0, size=len(index))
    data["AH"] = rng.uniform(0.5, 1.5, size=len(index))
    data[TARGET] = rng.uniform(0.5, 5.0, size=len(index))

    for col in FORBIDDEN_COLS:
        data[col] = rng.uniform(10.0, 50.0, size=len(index))

    df = pd.DataFrame(data, index=index)
    # Simulate realistic sensor outage blocks (NaNs in all input columns)
    # Block 1: hours 15 to 22 (8 hours)
    df.iloc[15:23, df.columns.isin(INPUT_COLS)] = np.nan
    # Block 2: hours 70 to 85 (16 hours)
    df.iloc[70:86, df.columns.isin(INPUT_COLS)] = np.nan

    return df


def test_build_features_returns_expected_structure(synthetic_hourly_grid):
    df = synthetic_hourly_grid
    feats = build_features(df, groups="all")

    assert isinstance(feats, pd.DataFrame)
    assert len(feats) == len(df)
    assert feats.index.equals(df.index)
    assert feats.index.name == "timestamp"

    # Default groups: time + lags
    default_feats = build_features(df)
    assert len(default_feats.columns) == 21  # 6 time + 15 lags
    assert "hour_sin" in default_feats.columns
    assert "PT08.S1(CO)_lag_1h" in default_feats.columns
    assert "PT08.S1(CO)_roll_mean_6h" not in default_feats.columns


def test_group_selection_and_validation(synthetic_hourly_grid):
    df = synthetic_hourly_grid

    # Single group as string
    time_feats = build_features(df, groups="time")
    assert list(time_feats.columns) == ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "dayofweek", "is_weekend"]

    # Subset sequence
    subset_feats = build_features(df, groups=["time", "env_interaction"])
    assert "hour_sin" in subset_feats.columns
    assert "T_x_RH" in subset_feats.columns
    assert "PT08.S1(CO)_lag_1h" not in subset_feats.columns

    # Unknown group raises ValueError
    with pytest.raises(ValueError, match="Unknown feature group"):
        build_features(df, groups=["time", "invalid_group"])


@pytest.mark.parametrize(
    "t_idx",
    [3, 18, 23, 24, 50, 75, 86, 115],
)
@pytest.mark.parametrize("perturbation_type", ["extreme", "nan"])
def test_no_temporal_leakage_future_to_past(synthetic_hourly_grid, t_idx, perturbation_type):
    """acceptance criterion 1: modifying inputs at t + k (k >= 1) must NEVER change features at t.

    Tests across multiple time regimes:
    - Early boundary (t=3)
    - Inside outage block 1 (t=18, 23)
    - Immediately after outage block 1 (t=24)
    - Normal contiguous operation (t=50)
    - Inside longer outage block 2 (t=75)
    - Immediately after outage block 2 (t=86)
    - Late boundary near horizon (t=115)
    With both extreme numerical perturbations and NaN injections in future slices.
    """
    df_clean = synthetic_hourly_grid
    df_perturbed = df_clean.copy()
    t_time = df_clean.index[t_idx]

    rng = np.random.default_rng(999 + t_idx)
    if perturbation_type == "extreme":
        for col in INPUT_COLS:
            df_perturbed.iloc[t_idx + 1 :, df_perturbed.columns.get_loc(col)] = rng.normal(
                9999.0, 500.0, size=len(df_clean) - t_idx - 1
            )
    elif perturbation_type == "nan":
        df_perturbed.iloc[t_idx + 1 :, df_perturbed.columns.isin(INPUT_COLS)] = np.nan

    feats_clean = build_features(df_clean, groups="all")
    feats_perturbed = build_features(df_perturbed, groups="all")

    # Features at all times <= t must be mathematically IDENTICAL
    past_clean = feats_clean.loc[:t_time]
    past_perturbed = feats_perturbed.loc[:t_time]

    pd.testing.assert_frame_equal(past_clean, past_perturbed)


def test_past_only_rolling_excludes_current_hour(synthetic_hourly_grid):
    """Verify shift(1) before rolling: modifying row t does not affect rolling features at t."""
    df_clean = synthetic_hourly_grid
    df_perturbed = df_clean.copy()

    t_idx = 50
    t_time = df_clean.index[t_idx]

    # Change sensor value ONLY at time t
    df_perturbed.loc[t_time, "PT08.S1(CO)"] += 5000.0

    roll_clean = build_features(df_clean, groups="rolling")
    roll_perturbed = build_features(df_perturbed, groups="rolling")

    # At time t, past rolling summary must NOT be altered
    pd.testing.assert_series_equal(roll_clean.loc[t_time], roll_perturbed.loc[t_time])

    # But at time t + 1, the perturbed value SHOULD be reflected in rolling summary
    t_next = df_clean.index[t_idx + 1]
    assert roll_clean.loc[t_next, "PT08.S1(CO)_roll_mean_6h"] != roll_perturbed.loc[t_next, "PT08.S1(CO)_roll_mean_6h"]


def test_no_target_or_deployment_leakage(synthetic_hourly_grid):
    """acceptance criterion 2: no feature derived from FORBIDDEN_COLS or TARGET (current or past)."""
    df_clean = synthetic_hourly_grid
    df_target_perturbed = df_clean.copy()

    # Modify TARGET and FORBIDDEN_COLS everywhere
    df_target_perturbed[TARGET] = 999.99
    for col in FORBIDDEN_COLS:
        df_target_perturbed[col] = 888.88

    feats_clean = build_features(df_clean, groups="all")
    feats_perturbed = build_features(df_target_perturbed, groups="all")

    # Features must be identical despite changes to target and forbidden columns
    pd.testing.assert_frame_equal(feats_clean, feats_perturbed)

    # Check column names: none should reference target or forbidden columns
    for col in feats_clean.columns:
        assert TARGET not in col
        for f_col in FORBIDDEN_COLS:
            assert f_col not in col


def test_gap_integrity_lags_do_not_bridge_gaps():
    """acceptance criterion 3: lags across unobserved hours produce NaN, not previous rows' values."""
    # Timeline with missing gaps:
    # Day 1: 10:00, 12:00 (11:00 is missing)
    # Day 2: 10:00 (24h after Day 1 10:00), 11:00 (24h after Day 1 11:00, which was missing), 12:00 (24h after Day 1 12:00)
    times = pd.to_datetime([
        "2004-05-01 10:00:00",
        "2004-05-01 12:00:00",  # 1h earlier (11:00) missing from grid
        "2004-05-02 10:00:00",  # 24h earlier (Day 1 10:00) exists
        "2004-05-02 11:00:00",  # 24h earlier (Day 1 11:00) missing from grid
        "2004-05-02 12:00:00",  # 24h earlier (Day 1 12:00) exists
    ])
    df_gap = pd.DataFrame(
        {col: [100.0, 200.0, 300.0, 400.0, 500.0] for col in SENSOR_COLS},
        index=times,
    )
    df_gap["T"] = 20.0
    df_gap["RH"] = 50.0
    df_gap["AH"] = 1.0

    feats = build_features(df_gap, groups="lags")

    # 1h lag assertions:
    # At 2004-05-01 12:00, 1h earlier would be 11:00 (which is unobserved in the grid).
    # It MUST yield NaN, not silently bridge to 10:00 (100.0)!
    assert np.isnan(feats.loc["2004-05-01 12:00:00", "PT08.S1(CO)_lag_1h"])
    # At 2004-05-02 11:00, 1h earlier is 10:00 (present with value 300.0)
    assert feats.loc["2004-05-02 11:00:00", "PT08.S1(CO)_lag_1h"] == 300.0

    # 24h lag assertions:
    # At 2004-05-02 10:00, 24h lag is 2004-05-01 10:00 (present with value 100.0)
    assert feats.loc["2004-05-02 10:00:00", "PT08.S1(CO)_lag_24h"] == 100.0
    # At 2004-05-02 11:00, 24h earlier is 2004-05-01 11:00 (unobserved in the grid).
    # It MUST yield NaN!
    assert np.isnan(feats.loc["2004-05-02 11:00:00", "PT08.S1(CO)_lag_24h"])
    # At 2004-05-02 12:00, 24h lag is 2004-05-01 12:00 (present with value 200.0)
    assert feats.loc["2004-05-02 12:00:00", "PT08.S1(CO)_lag_24h"] == 200.0


def test_longest_lookback_constant():
    assert LONGEST_LOOKBACK_HOURS == 24
