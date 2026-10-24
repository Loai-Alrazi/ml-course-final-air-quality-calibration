"""Tests for modeling pipelines, cross-validation runner, and final-test isolation.

Implements acceptance criteria for Issue #8 (part of #4):
1. Imputer and scaler fitted on training-fold rows only (exercised through the runner).
2. Final-test values do not affect design matrix or CV predictions; all CV timestamps < TEST_START.
3. Equivalent evaluation sets: identical validation timestamps across all experiments per fold.
4. Final-test path refuses without authorization and works on synthetic data.
5. Runner and pipeline interfaces adhere to contract and project specifications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from air_quality.data import (
    DEFAULT_RAW_DIR,
    FORBIDDEN_COLS,
    INPUT_COLS,
    SENSOR_COLS,
    TARGET,
    eligible_mask,
    load_raw,
)
from air_quality.features import (
    FEATURE_GROUPS,
)
import air_quality.modeling as modeling
from air_quality.modeling import (
    HGB_SETTINGS,
    PREDICTION_COLUMNS,
    RIDGE_ALPHA,
    build_design_matrix,
    format_feature_set,
    make_pipeline,
    parse_d_groups,
    predict_final_test,
    run_all_experiments,
    run_cv_experiment,
)
from air_quality.split import (
    TEST_START,
    cv_indices,
    dev_folds,
    split_dev_test,
)


@pytest.fixture
def synthetic_full_grid() -> pd.DataFrame:
    """Create a synthetic hourly grid spanning across development and final test periods (encompassing TEST_START)."""
    start = pd.Timestamp("2004-03-10 18:00:00")
    end = pd.Timestamp("2005-02-28 23:00:00")
    index = pd.date_range(start, end, freq="h", name="timestamp")
    rng = np.random.default_rng(42)

    data = {col: rng.normal(1000.0, 150.0, size=len(index)) for col in SENSOR_COLS}
    data["T"] = rng.uniform(5.0, 35.0, size=len(index))
    data["RH"] = rng.uniform(20.0, 85.0, size=len(index))
    data["AH"] = rng.uniform(0.5, 2.0, size=len(index))
    data[TARGET] = rng.uniform(0.5, 6.0, size=len(index))

    for col in FORBIDDEN_COLS:
        data[col] = rng.uniform(10.0, 100.0, size=len(index))

    df = pd.DataFrame(data, index=index)

    # Inject simulated missing blocks (NaNs in inputs) to test robustness
    df.iloc[100:110, df.columns.isin(INPUT_COLS)] = np.nan
    df.iloc[500:520, df.columns.isin(INPUT_COLS)] = np.nan
    return df


def test_pipeline_construction_and_fixed_settings():
    """Verify that pipelines are created with fixed settings and no search."""
    dummy_pipe = make_pipeline("dummy_median")
    assert "regressor" in dummy_pipe.named_steps
    assert dummy_pipe.named_steps["regressor"].strategy == "median"

    ridge_pipe = make_pipeline("ridge")
    assert "imputer" in ridge_pipe.named_steps
    assert "scaler" in ridge_pipe.named_steps
    assert "regressor" in ridge_pipe.named_steps
    assert ridge_pipe.named_steps["imputer"].strategy == "median"
    assert ridge_pipe.named_steps["regressor"].alpha == RIDGE_ALPHA
    assert ridge_pipe.named_steps["regressor"].random_state == 42

    hgb_pipe = make_pipeline("hgb")
    assert "regressor" in hgb_pipe.named_steps
    reg = hgb_pipe.named_steps["regressor"]
    for key, val in HGB_SETTINGS.items():
        assert getattr(reg, key) == val

    with pytest.raises(ValueError, match="Unknown model_type"):
        make_pipeline("unsupported_model")


def test_feature_set_formatting_and_d_groups_parsing():
    """Verify canonical feature_set naming and CLI flag parsing."""
    assert format_feature_set(None) == "inputs"
    assert format_feature_set([]) == "inputs"
    assert format_feature_set(["lags", "time"]) == "inputs+time+lags"
    assert format_feature_set(["missing_ind"]) == "inputs+missing_ind"

    assert parse_d_groups(None) is None
    assert parse_d_groups("") == []
    assert parse_d_groups("none") == []
    assert parse_d_groups("time, lags") == ["time", "lags"]
    assert parse_d_groups(["time", "rolling"]) == ["time", "rolling"]

    with pytest.raises(ValueError, match="Unknown feature group"):
        parse_d_groups("time,invalid_group")


class FitRecorder(BaseEstimator, TransformerMixin):
    calls: list[pd.DatetimeIndex] = []

    def fit(self, X, y=None):
        FitRecorder.calls.append(pd.DatetimeIndex(X.index))
        return self

    def transform(self, X):
        return X


def test_runner_fits_on_training_rows_only(synthetic_full_grid, monkeypatch):
    """Acceptance criterion 1: verify that the runner passes training rows only to fit.

    Prepends a recording step to every pipeline the runner builds, runs all experiments,
    and asserts that every fitted index precedes validation start and never overlaps validation rows.
    """
    orig = modeling.make_pipeline
    monkeypatch.setattr(
        modeling,
        "make_pipeline",
        lambda t: Pipeline([("record", FitRecorder()), *orig(t).steps]),
    )
    FitRecorder.calls.clear()
    preds = modeling.run_all_experiments(synthetic_full_grid, d_groups=list(FEATURE_GROUPS))
    folds = dev_folds()
    assert len(FitRecorder.calls) == preds["experiment"].nunique() * len(folds)
    for i, fitted in enumerate(FitRecorder.calls):
        k = i % len(folds)
        val = preds.loc[preds["fold"] == str(k), "timestamp"]
        assert fitted.max() < folds[k].val_start
        assert not fitted.isin(val).any()


def test_fold_local_imputer_and_scaler(synthetic_full_grid):
    """Acceptance criterion 1 (unit level): imputers and scalers are fitted on training rows only.

    Perturbing validation-fold rows leaves the fitted imputer and scaler statistics unchanged.
    """
    df = synthetic_full_grid
    X, y = build_design_matrix(df, groups=None, base="inputs")
    cv = cv_indices(X.index, dev_folds())

    pipe = make_pipeline("ridge")

    # Fit each fold on clean data and on data where validation rows are perturbed
    for train_idx, val_idx in cv:
        X_clean = X.copy()
        X_pert = X.copy()

        # Aggressively perturb validation rows
        X_pert.iloc[val_idx] = X_pert.iloc[val_idx] * 1000.0 + 99999.0

        clean_fitted = clone(pipe).fit(X_clean.iloc[train_idx], y.iloc[train_idx])
        pert_fitted = clone(pipe).fit(X_pert.iloc[train_idx], y.iloc[train_idx])

        # Imputer statistics (medians) must be mathematically identical
        np.testing.assert_array_equal(
            clean_fitted.named_steps["imputer"].statistics_,
            pert_fitted.named_steps["imputer"].statistics_,
        )

        # Scaler statistics (mean and scale) must be mathematically identical
        np.testing.assert_array_equal(
            clean_fitted.named_steps["scaler"].mean_,
            pert_fitted.named_steps["scaler"].mean_,
        )
        np.testing.assert_array_equal(
            clean_fitted.named_steps["scaler"].scale_,
            pert_fitted.named_steps["scaler"].scale_,
        )


def test_final_test_isolation_and_no_cv_timestamp_exceeds_test_start(synthetic_full_grid):
    """Acceptance criterion 2: altering final-test data leaves design matrix and CV unchanged.

    Covers lags, rolling, and all 10 experiments across clean and perturbed grids.
    Also asserts that no CV timestamp is >= TEST_START.
    """
    df_clean = synthetic_full_grid
    df_perturbed = df_clean.copy()

    # Modify all values in the final-test period
    test_rows = df_perturbed.index >= TEST_START
    assert test_rows.any()

    # Set all sensor and target values in test period to extreme values and NaNs
    df_perturbed.loc[test_rows, INPUT_COLS] = 999999.0
    df_perturbed.loc[test_rows, TARGET] = np.nan
    for col in FORBIDDEN_COLS:
        df_perturbed.loc[test_rows, col] = 888888.0

    # 1. Design matrix must be identical across all feature groups (including lags and rolling)
    X_clean, y_clean = build_design_matrix(df_clean, groups="all")
    X_pert, y_pert = build_design_matrix(df_perturbed, groups="all")

    pd.testing.assert_frame_equal(X_clean, X_pert)
    pd.testing.assert_series_equal(y_clean, y_pert)

    # 2. CV predictions must be identical across all 10 experiments
    preds_clean = run_all_experiments(df_clean, d_groups=list(FEATURE_GROUPS))
    preds_pert = run_all_experiments(df_perturbed, d_groups=list(FEATURE_GROUPS))

    pd.testing.assert_frame_equal(preds_clean, preds_pert)

    # 3. All CV timestamps must strictly precede TEST_START
    assert (preds_clean["timestamp"] < TEST_START).all()
    assert (preds_pert["timestamp"] < TEST_START).all()


def test_identical_validation_timestamps_across_experiments(synthetic_full_grid):
    """Acceptance criterion 3: all experiments yield identical validation timestamps per fold."""
    df = synthetic_full_grid
    preds = run_all_experiments(df, d_groups=["time", "lags"])

    experiments = preds["experiment"].unique().tolist()
    assert "A" in experiments
    assert "B" in experiments
    assert "C" in experiments
    assert "C+time" in experiments
    assert "D" in experiments
    assert "E" in experiments

    # Check across each fold
    for fold in ["0", "1", "2"]:
        ref_ts = None
        for exp in experiments:
            exp_ts = preds.loc[
                (preds["experiment"] == exp) & (preds["fold"] == fold),
                "timestamp",
            ].reset_index(drop=True)

            assert len(exp_ts) > 0, f"No rows for experiment {exp} in fold {fold}"
            if ref_ts is None:
                ref_ts = exp_ts
            else:
                pd.testing.assert_series_equal(
                    exp_ts,
                    ref_ts,
                    check_names=False,
                    obj=f"Timestamps mismatch between experiment {exp} and reference in fold {fold}",
                )


def test_final_test_path_locked_and_synthetic_evaluation(synthetic_full_grid):
    """Acceptance criterion 4: final-test path refuses without authorization and works on synthetic data."""
    df = synthetic_full_grid

    # Refuse without authorization
    with pytest.raises(PermissionError, match="authorized=True"):
        predict_final_test(df, "C")

    with pytest.raises(PermissionError, match="authorized=True"):
        predict_final_test(df, "C", authorized=False)

    # Execute with explicit authorization
    test_preds = predict_final_test(df, "C", authorized=True)

    assert isinstance(test_preds, pd.DataFrame)
    assert list(test_preds.columns) == PREDICTION_COLUMNS
    assert (test_preds["fold"] == "test").all()
    assert (test_preds["timestamp"] >= TEST_START).all()
    assert len(test_preds) > 0
    assert not test_preds["y_pred"].isna().any()

    # Also test with experiment E (HistGradientBoosting)
    test_preds_e = predict_final_test(df, "E", authorized=True, d_groups=["time"])
    assert (test_preds_e["model"] == "hgb").all()
    assert len(test_preds_e) == len(test_preds)
    assert not test_preds_e["y_pred"].isna().any()


def test_design_matrix_preserves_nan_features(synthetic_full_grid):
    """Verify that design matrix never drops rows due to NaN features."""
    df = synthetic_full_grid
    # lag_24h will have NaNs in the first 24 hours
    X, y = build_design_matrix(df, groups="lags", base="inputs")

    dev_df, _ = split_dev_test(df)
    expected_rows = eligible_mask(dev_df).sum()

    assert len(X) == expected_rows
    assert len(y) == expected_rows
    # Confirm that X contains NaNs (e.g. from early lags or outage blocks)
    assert X.isna().any().any()


def test_run_all_experiments_structure_and_counts(synthetic_full_grid):
    """Verify output structure and column types for run_all_experiments."""
    df = synthetic_full_grid

    # Run without --d-groups: 8 experiments
    res8 = run_all_experiments(df, d_groups=None)
    assert res8["experiment"].nunique() == 8
    assert set(res8["experiment"].unique()) == {
        "A",
        "B",
        "C",
        "C+time",
        "C+lags",
        "C+rolling",
        "C+env_interaction",
        "C+missing_ind",
    }
    assert list(res8.columns) == PREDICTION_COLUMNS
    assert not res8["y_pred"].isna().any()

    # Run with empty d_groups: 10 experiments, D and E use inputs
    res10 = run_all_experiments(df, d_groups=[])
    assert res10["experiment"].nunique() == 10
    assert set(res10["experiment"].unique()) == {
        "A",
        "B",
        "C",
        "C+time",
        "C+lags",
        "C+rolling",
        "C+env_interaction",
        "C+missing_ind",
        "D",
        "E",
    }
    d_rows = res10[res10["experiment"] == "D"]
    assert (d_rows["feature_set"] == "inputs").all()
    e_rows = res10[res10["experiment"] == "E"]
    assert (e_rows["feature_set"] == "inputs").all()
    assert (e_rows["model"] == "hgb").all()


@pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / "AirQualityUCI.csv").exists(),
    reason="Real CSV not available in test environment",
)
def test_real_data_cv_predictions_invariants():
    """Invariants check on real dataset: 2,711 rows per experiment, no NaNs, correct folds."""
    df = load_raw()
    preds = run_all_experiments(df, d_groups=None)

    assert len(preds) == 21688  # 8 experiments * 2,711 rows
    assert preds["fold"].isin(["0", "1", "2"]).all()
    assert (preds["timestamp"] < TEST_START).all()
    assert not preds["y_pred"].isna().any()

    # Check each fold count
    fold_counts = preds[preds["experiment"] == "A"]["fold"].value_counts().to_dict()
    assert fold_counts == {"0": 771, "1": 936, "2": 1004}
