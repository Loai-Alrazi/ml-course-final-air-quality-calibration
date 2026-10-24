"""Modeling pipelines and cross-validation experiment runner for CO sensor calibration.

Implements acceptance criteria for Issue #8 (part of #4):
1. Design matrix construction on complete hourly grid with fold-local imputation.
2. Fixed pipelines for baselines and models A–E.
3. Out-of-fold cross-validation predictions on primary development folds.
4. Locked final-test path requiring explicit authorization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from air_quality.data import (
    INPUT_COLS,
    PROJECT_ROOT,
    SENSOR_COLS,
    TARGET,
    eligible_mask,
    load_raw,
)
from air_quality.features import (
    FEATURE_GROUPS,
    build_features,
)
from air_quality.split import (
    TEST_START,
    cv_indices,
    dev_folds,
    split_dev_test,
)

# Output directory and default filename
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "cv_predictions.csv"

# Fixed model settings (no search)
RIDGE_ALPHA = 1.0
HGB_SETTINGS = {
    "max_iter": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "random_state": 42,
}

# Standard column schema for shared predictions file
PREDICTION_COLUMNS = [
    "experiment",
    "model",
    "feature_set",
    "fold",
    "timestamp",
    "y_true",
    "y_pred",
]

# Supported experiments
BASE_EXPERIMENTS = ("A", "B", "C")
CANDIDATE_EXPERIMENTS = tuple(f"C+{g}" for g in FEATURE_GROUPS)
EXTENDED_EXPERIMENTS = ("D", "E")
ALL_EXPERIMENTS = BASE_EXPERIMENTS + CANDIDATE_EXPERIMENTS + EXTENDED_EXPERIMENTS


def make_pipeline(model_type: str) -> Pipeline:
    """Create a modeling pipeline with fixed settings and no hyperparameter search.

    Parameters
    ----------
    model_type : str
        One of 'dummy_median', 'ridge', or 'hgb'.

    Returns
    -------
    Pipeline
        Configured scikit-learn pipeline.
    """
    if model_type == "dummy_median":
        return Pipeline([
            ("regressor", DummyRegressor(strategy="median")),
        ])
    elif model_type == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=RIDGE_ALPHA, random_state=42)),
        ])
    elif model_type == "hgb":
        return Pipeline([
            ("regressor", HistGradientBoostingRegressor(**HGB_SETTINGS)),
        ])
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Supported types: 'dummy_median', 'ridge', 'hgb'."
        )


def format_feature_set(groups: Sequence[str] | None) -> str:
    """Format canonical feature_set string from a sequence of group names."""
    if not groups:
        return "inputs"
    # Order according to canonical FEATURE_GROUPS order
    ordered = [g for g in FEATURE_GROUPS if g in groups]
    for g in groups:
        if g not in ordered:
            ordered.append(g)
    return "inputs+" + "+".join(ordered)


def parse_d_groups(d_groups_arg: str | Sequence[str] | None) -> list[str] | None:
    """Parse --d-groups CLI argument into a list of feature group names.

    Returns None if flag was not provided. Returns an empty list if flag was
    provided as empty string or 'none'.
    """
    if d_groups_arg is None:
        return None
    if isinstance(d_groups_arg, str):
        cleaned = d_groups_arg.strip()
        if not cleaned or cleaned.lower() == "none":
            return []
        items = [item.strip() for item in cleaned.split(",") if item.strip()]
    else:
        items = [str(g).strip() for g in d_groups_arg if str(g).strip()]

    # Validate group names
    for g in items:
        if g not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group {g!r} in --d-groups. Supported: {FEATURE_GROUPS}"
            )
    return items


def build_design_matrix(
    df: pd.DataFrame,
    groups: Sequence[str] | str | None = None,
    base: str = "inputs",
) -> tuple[pd.DataFrame, pd.Series]:
    """Build design matrix X and target y for the development period.

    Follows contract §10:
    1. Features are built on the complete hourly grid before splitting/filtering.
    2. Temporal split via split_dev_test separates development and test rows.
    3. Row filtering via eligible_mask applies only after splitting.
    4. Missing values in features are never dropped; they are handled inside pipelines.

    Parameters
    ----------
    df : pd.DataFrame
        Complete hourly grid returned by load_raw().
    groups : Sequence[str] | str | None, optional
        Feature groups to build via build_features.
    base : str, default 'inputs'
        Base feature columns: 'S1' (['PT08.S1(CO)']) or 'inputs' (INPUT_COLS).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Development feature matrix X and target series y.
    """
    if base == "S1":
        base_cols = ["PT08.S1(CO)"]
    elif base == "inputs":
        base_cols = list(INPUT_COLS)
    else:
        raise ValueError(f"Unknown base feature set: {base!r}. Choose 'S1' or 'inputs'.")

    # Base inputs from raw dataframe
    base_df = df[base_cols]

    # Add engineered features if requested
    if groups:
        if isinstance(groups, str) and groups != "all":
            selected_groups = [groups]
        else:
            selected_groups = groups
        feats = build_features(df, groups=selected_groups)
        full_features = base_df.join(feats)
    else:
        full_features = base_df

    # Chronological split
    dev_X, _ = split_dev_test(full_features)
    dev_df, _ = split_dev_test(df)

    # Apply row eligibility filter to development period
    mask = eligible_mask(dev_df)
    X = dev_X[mask]
    y = dev_df.loc[mask, TARGET]

    return X, y


def _resolve_experiment_spec(
    experiment: str,
    d_groups: Sequence[str] | None = None,
) -> tuple[str, str, str, list[str] | None]:
    """Resolve model_type, feature_set, base, and groups for a given experiment name."""
    if experiment == "A":
        return "dummy_median", "S1", "S1", None
    elif experiment == "B":
        return "ridge", "S1", "S1", None
    elif experiment == "C":
        return "ridge", "inputs", "inputs", None
    elif experiment.startswith("C+"):
        group = experiment[len("C+") :]
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown candidate group {group!r} in experiment {experiment!r}.")
        return "ridge", f"inputs+{group}", "inputs", [group]
    elif experiment == "D":
        groups = list(d_groups) if d_groups is not None else []
        feat_set = format_feature_set(groups)
        return "ridge", feat_set, "inputs", groups
    elif experiment == "E":
        groups = list(d_groups) if d_groups is not None else []
        feat_set = format_feature_set(groups)
        return "hgb", feat_set, "inputs", groups
    else:
        raise ValueError(f"Unknown experiment name: {experiment!r}.")


def run_cv_experiment(
    df: pd.DataFrame,
    experiment: str,
    d_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run cross-validation on primary development folds for a single experiment.

    Uses cross_validate(pipe, X, y, cv=cv_indices(X.index, dev_folds()), return_estimator=True, return_indices=True)
    and predicts validation rows with each fold's fitted estimator.

    Parameters
    ----------
    df : pd.DataFrame
        Complete hourly grid.
    experiment : str
        Experiment name.
    d_groups : Sequence[str] | None, optional
        Retained groups if experiment is 'D' or 'E'.

    Returns
    -------
    pd.DataFrame
        Out-of-fold predictions with standard columns.
    """
    model_type, feat_set, base, groups = _resolve_experiment_spec(experiment, d_groups)
    X, y = build_design_matrix(df, groups=groups, base=base)

    folds = dev_folds()
    cv = cv_indices(X.index, folds)

    pipe = make_pipeline(model_type)
    res = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        return_estimator=True,
        return_indices=True,
    )

    rows: list[pd.DataFrame] = []
    for f_idx, est in enumerate(res["estimator"]):
        val_idx = res["indices"]["test"][f_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        y_pred = est.predict(X_val)

        sub_df = pd.DataFrame({
            "experiment": experiment,
            "model": model_type,
            "feature_set": feat_set,
            "fold": str(f_idx),
            "timestamp": X_val.index,
            "y_true": y_val.to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        })
        rows.append(sub_df)

    return pd.concat(rows, ignore_index=True)


def run_all_experiments(
    df: pd.DataFrame,
    d_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run cross-validation across all active experiments.

    Without d_groups (d_groups is None): runs A, B, C and C+<group> (8 experiments).
    With d_groups (list or empty list): also runs D and E (10 experiments total).

    Parameters
    ----------
    df : pd.DataFrame
        Complete hourly grid.
    d_groups : Sequence[str] | None, optional
        Feature groups retained for experiments D and E.

    Returns
    -------
    pd.DataFrame
        Combined out-of-fold predictions table matching shared format.
    """
    # Base experiments A, B, C
    experiments_to_run = list(BASE_EXPERIMENTS)

    # 5 ablation candidate experiments
    experiments_to_run.extend(CANDIDATE_EXPERIMENTS)

    # If --d-groups was passed (even if empty list), include D and E
    if d_groups is not None:
        experiments_to_run.extend(EXTENDED_EXPERIMENTS)

    # Optimize feature generation: precompute full features once
    all_engineered = build_features(df, groups="all")
    full_feature_df = df[INPUT_COLS].join(all_engineered)

    dev_full, _ = split_dev_test(full_feature_df)
    dev_raw, _ = split_dev_test(df)

    mask = eligible_mask(dev_raw)
    dev_usable = dev_full[mask]
    y_dev = dev_raw.loc[mask, TARGET]

    folds = dev_folds()
    cv = cv_indices(dev_usable.index, folds)

    all_rows: list[pd.DataFrame] = []

    for exp_name in experiments_to_run:
        model_type, feat_set, base, groups = _resolve_experiment_spec(exp_name, d_groups)

        # Slice exact columns needed for this experiment
        if base == "S1":
            exp_cols = ["PT08.S1(CO)"]
        else:
            if not groups:
                exp_cols = list(INPUT_COLS)
            else:
                eng_cols = build_features(df.iloc[:2], groups=groups).columns.tolist()
                exp_cols = list(INPUT_COLS) + eng_cols

        X_exp = dev_usable[exp_cols]
        pipe = make_pipeline(model_type)

        res = cross_validate(
            pipe,
            X_exp,
            y_dev,
            cv=cv,
            return_estimator=True,
            return_indices=True,
        )

        for f_idx, est in enumerate(res["estimator"]):
            val_idx = res["indices"]["test"][f_idx]
            X_val = X_exp.iloc[val_idx]
            y_val = y_dev.iloc[val_idx]
            y_pred = est.predict(X_val)

            sub_df = pd.DataFrame({
                "experiment": exp_name,
                "model": model_type,
                "feature_set": feat_set,
                "fold": str(f_idx),
                "timestamp": X_val.index,
                "y_true": y_val.to_numpy(dtype=float),
                "y_pred": np.asarray(y_pred, dtype=float),
            })
            all_rows.append(sub_df)

    out = pd.concat(all_rows, ignore_index=True)
    return out[PREDICTION_COLUMNS]


def predict_final_test(
    df: pd.DataFrame,
    experiment: str,
    *,
    authorized: bool = False,
    d_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Refit a given experiment on all eligible development rows and predict final-test rows.

    This function implements the locked final-test evaluation path. It must refuse to
    run without an explicit authorization argument (authorized=True).

    Parameters
    ----------
    df : pd.DataFrame
        Complete hourly grid spanning development and test periods.
    experiment : str
        Experiment to evaluate.
    authorized : bool, default False
        Explicit authorization flag. Must be True to execute.
    d_groups : Sequence[str] | None, optional
        Retained groups if experiment is 'D' or 'E'.

    Returns
    -------
    pd.DataFrame
        Predictions on eligible final-test rows with fold='test'.

    Raises
    ------
    PermissionError
        If authorized is False.
    """
    if not authorized:
        raise PermissionError(
            "Final-test evaluation is locked to prevent premature evaluation and leakage. "
            "Call predict_final_test with authorized=True to run."
        )

    model_type, feat_set, base, groups = _resolve_experiment_spec(experiment, d_groups)

    # Base columns
    if base == "S1":
        base_cols = ["PT08.S1(CO)"]
    else:
        base_cols = list(INPUT_COLS)

    base_df = df[base_cols]
    if groups:
        feats = build_features(df, groups=groups)
        full_X = base_df.join(feats)
    else:
        full_X = base_df

    # Split into development and final test
    dev_X, test_X = split_dev_test(full_X)
    dev_df, test_df = split_dev_test(df)

    # Eligible masks
    dev_mask = eligible_mask(dev_df)
    test_mask = eligible_mask(test_df)

    X_train = dev_X[dev_mask]
    y_train = dev_df.loc[dev_mask, TARGET]

    X_test = test_X[test_mask]
    y_test = test_df.loc[test_mask, TARGET]

    # Fit on all eligible development data
    pipe = make_pipeline(model_type)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    out = pd.DataFrame({
        "experiment": experiment,
        "model": model_type,
        "feature_set": feat_set,
        "fold": "test",
        "timestamp": test_df.index[test_mask],
        "y_true": y_test.to_numpy(dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    return out[PREDICTION_COLUMNS]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI runner to execute cross-validation experiments and write cv_predictions.csv."""
    parser = argparse.ArgumentParser(
        description="Run cross-validation modeling experiments for CO sensor calibration."
    )
    parser.add_argument(
        "--d-groups",
        type=str,
        default=None,
        help="Comma-separated feature groups retained for experiments D and E (e.g. 'time,lags' or ''). "
        "If omitted, only experiments A, B, C and C+<group> are run.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to write predictions CSV (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Optional path to raw CSV dataset.",
    )

    args = parser.parse_args(argv)

    d_groups = parse_d_groups(args.d_groups)

    print("Loading raw hourly data...")
    t_start = time.time()
    df = load_raw(args.raw_path)

    print("Running cross-validation experiments...")
    if d_groups is None:
        print("Running 8 experiments: A, B, C, and 5 candidate groups C+<group>.")
    else:
        print(
            f"Running 10 experiments: A, B, C, 5 candidates, plus D and E with groups: {d_groups or ['(none)']}"
        )

    preds = run_all_experiments(df, d_groups=d_groups)

    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)

    t_elapsed = time.time() - t_start
    print(f"Generated {len(preds)} predictions across {preds['experiment'].nunique()} experiments.")
    print(f"Saved to {output_path} in {t_elapsed:.2f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
