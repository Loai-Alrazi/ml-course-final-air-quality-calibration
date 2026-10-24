"""Tests for the evaluation of out-of-fold predictions (Issue #9).

Predictions are synthetic frames in the shared format of Issue #8, placed in
the real primary validation windows. No model is trained here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from air_quality import evaluation as ev
from air_quality.data import CSV_NAME, DEFAULT_RAW_DIR, INPUT_COLS, TARGET, load_raw
from air_quality.features import FEATURE_GROUPS
from air_quality.split import DATA_START, TEST_START, dev_folds

N_PER_FOLD = 6
FEATURES = {"A": "none", "B": "S1", "C": "inputs", **{f"C+{g}": f"inputs+{g}" for g in FEATURE_GROUPS}}


def make_preds(fold_errors: dict[str, list[float]], d_groups=()) -> pd.DataFrame:
    """Rows whose absolute error is fold_errors[exp][fold], so fold MAE equals that value."""
    rows = []
    for exp, errors in fold_errors.items():
        feature_set = FEATURES.get(exp, ev.d_feature_set(d_groups))
        for k, (fold, err) in enumerate(zip(dev_folds(), errors)):
            for h in range(N_PER_FOLD):
                y = 1.0 + 0.5 * h + k
                rows.append({
                    "experiment": exp, "model": ev.EXPECTED_MODEL[exp], "feature_set": feature_set,
                    "fold": str(k), "timestamp": fold.val_start + pd.Timedelta(hours=h),
                    "y_true": y, "y_pred": y + (err if h % 2 else -err),
                })
    return pd.DataFrame(rows, columns=ev.PREDICTION_COLS)


def stage1(c=(1.0, 1.0, 1.0), **candidates) -> dict[str, list[float]]:
    errors = {"A": [3.0] * 3, "B": [2.0] * 3, "C": list(c)}
    for g in FEATURE_GROUPS:
        errors[f"C+{g}"] = list(candidates.get(g, (1.5, 1.5, 1.5)))
    return errors


@pytest.fixture
def valid() -> pd.DataFrame:
    return make_preds(stage1())


def grid_frame(start=DATA_START, end="2005-04-04 14:00") -> pd.DataFrame:
    index = pd.date_range(start, end, freq="h", name="timestamp")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(1000, 100, size=(len(index), len(INPUT_COLS))), index=index, columns=INPUT_COLS)
    df["T"] = rng.uniform(0, 30, len(index))
    df["RH"] = rng.uniform(20, 80, len(index))
    df[TARGET] = rng.uniform(0.1, 8, len(index))
    return df


# --- Validation of the shared format --------------------------------------


def test_valid_frame_round_trips_through_csv(valid, tmp_path):
    path = tmp_path / "cv_predictions.csv"
    valid.to_csv(path, index=False)
    loaded = ev.load_predictions(path)
    pd.testing.assert_frame_equal(loaded, ev.validate_predictions(valid))
    assert set(loaded["fold"]) == {"0", "1", "2"}


def test_integer_fold_labels_are_normalised(valid):
    as_int = valid.assign(fold=valid["fold"].astype(int))
    assert ev.validate_predictions(as_int)["fold"].tolist() == ev.validate_predictions(valid)["fold"].tolist()


@pytest.mark.parametrize("change", ["drop", "extra"])
def test_rejects_missing_or_extra_columns(valid, change):
    bad = valid.drop(columns="y_pred") if change == "drop" else valid.assign(note="x")
    with pytest.raises(ValueError, match="columns"):
        ev.validate_predictions(bad)


def test_rejects_missing_values(valid):
    valid.loc[3, "y_pred"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        ev.validate_predictions(valid)


def test_rejects_unknown_experiment_and_fold(valid):
    with pytest.raises(ValueError, match="unknown experiments"):
        ev.validate_predictions(valid.replace({"experiment": {"B": "B2"}}))
    with pytest.raises(ValueError, match="unexpected folds"):
        ev.validate_predictions(valid.replace({"fold": {"2": "3"}}))


def test_rejects_duplicate_rows(valid):
    with pytest.raises(ValueError, match="repeat"):
        ev.validate_predictions(pd.concat([valid, valid.iloc[[0]]]))


def test_rejects_unequal_evaluation_sets(valid):
    drop = valid.index[(valid["experiment"] == "C") & (valid["fold"] == "1")][0]
    with pytest.raises(ValueError, match="different hours"):
        ev.validate_predictions(valid.drop(index=drop))


def test_rejects_experiment_missing_a_fold(valid):
    bad = valid[~((valid["experiment"] == "B") & (valid["fold"] == "2"))]
    with pytest.raises(ValueError, match="lacks experiments"):
        ev.validate_predictions(bad)


def test_rejects_row_outside_its_fold_window(valid):
    valid.loc[valid["fold"] == "0", "fold"] = "1"     # mislabelled fold
    with pytest.raises(ValueError, match="outside its validation window"):
        ev.validate_predictions(valid)


def test_rejects_inconsistent_targets(valid):
    valid.loc[valid.index[valid["experiment"] == "A"][0], "y_true"] += 0.1
    with pytest.raises(ValueError, match="y_true differs"):
        ev.validate_predictions(valid)


def test_rejects_wrong_or_changing_design(valid):
    with pytest.raises(ValueError, match="expected 'ridge'"):
        ev.validate_predictions(valid.replace({"model": {"ridge": "hgb"}}))
    bad = valid.copy()
    bad.loc[bad.index[bad["experiment"] == "C"][0], "feature_set"] = "inputs+time"
    with pytest.raises(ValueError, match="changes within"):
        ev.validate_predictions(bad)
    with pytest.raises(ValueError, match="feature_set"):
        ev.validate_predictions(valid.replace({"feature_set": {"inputs+lags": "inputs+time"}}))


def test_final_test_rows_rejected_by_default(valid):
    test_rows = valid[valid["experiment"] == "C"].head(2).assign(
        fold="test", timestamp=[TEST_START, TEST_START + pd.Timedelta(hours=1)]
    )
    combined = pd.concat([valid, test_rows], ignore_index=True)
    with pytest.raises(ValueError, match="allow_final_test"):
        ev.validate_predictions(combined)
    assert (ev.validate_predictions(combined, allow_final_test=True)["fold"] == "test").sum() == 2

    cv_late = valid.copy()
    cv_late.loc[0, "timestamp"] = TEST_START
    with pytest.raises(ValueError, match="TEST_START"):
        ev.validate_predictions(cv_late)
    early = test_rows.assign(timestamp=[TEST_START - pd.Timedelta(hours=1), TEST_START])
    with pytest.raises(ValueError, match="before TEST_START"):
        ev.validate_predictions(pd.concat([valid, early], ignore_index=True), allow_final_test=True)


# --- Metrics --------------------------------------------------------------


def test_regression_metrics_hand_checked():
    m = ev.regression_metrics([1, 2, 3, 4], [1, 2, 3, 8])
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(2.0)
    assert m["r2"] == pytest.approx(1 - 16 / 5)


def test_cv_summary_mean_std_and_per_fold(valid):
    preds = ev.validate_predictions(make_preds(stage1(c=(1.0, 2.0, 3.0))))
    s = ev.cv_summary(preds)
    assert list(s.index[:3]) == ["A", "B", "C"]
    assert s.loc["C", "mae_mean"] == pytest.approx(2.0)
    assert s.loc["C", "mae_std"] == pytest.approx(1.0)       # ddof=1
    assert [s.loc["C", f"mae_fold_{k}"] for k in range(3)] == pytest.approx([1.0, 2.0, 3.0])
    assert (s["n_folds"] == 3).all()


# --- Selection rule and procedure -----------------------------------------


def rule(cand, ref):
    preds = ev.validate_predictions(make_preds({"B": list(ref), "C": list(cand)}))
    return ev.compare(preds, "C", "B")


def test_rule_rejects_lower_mean_with_only_one_fold_improved():
    c = rule(cand=(0.5, 1.1, 1.1), ref=(1.0, 1.0, 1.0))
    assert c.candidate_mae < c.reference_mae and c.folds_improved == 1 and not c.passes


def test_rule_rejects_majority_of_folds_with_higher_mean():
    c = rule(cand=(0.9, 0.9, 2.0), ref=(1.0, 1.0, 1.0))
    assert c.folds_improved == 2 and not c.passes


def test_rule_passes_lower_mean_and_two_folds():
    assert rule(cand=(0.8, 0.9, 1.05), ref=(1.0, 1.0, 1.0)).passes


def test_rule_counts_ties_as_no_improvement():
    c = rule(cand=(1.0, 1.0, 0.7), ref=(1.0, 1.0, 1.0))
    assert c.folds_improved == 1 and not c.passes


def test_ablation_retains_only_groups_that_pass():
    preds = ev.validate_predictions(make_preds(stage1(lags=(0.8, 0.9, 0.95), rolling=(0.5, 1.2, 1.2))))
    table = ev.ablation_table(preds)
    assert table["group"].tolist() == list(FEATURE_GROUPS)
    assert ev.retained_groups(preds) == ["lags"]
    assert table.set_index("group").loc["rolling", "folds_improved"] == "1/3"


def full(d, e, groups=("lags",), b=(2.0, 2.0, 2.0)) -> pd.DataFrame:
    errors = stage1(lags=(0.8, 0.9, 0.95))
    errors.update(B=list(b), D=list(d), E=list(e))
    return ev.validate_predictions(make_preds(errors, d_groups=groups))


@pytest.mark.parametrize(
    "d, e, expected",
    [
        ((0.8, 0.9, 0.95), (0.5, 0.6, 0.7), "E"),     # D beats C, E beats D
        ((0.8, 0.9, 0.95), (0.5, 1.0, 1.0), "D"),     # E better on average but in 1 fold only
        ((1.1, 1.1, 0.5), (0.9, 0.9, 0.9), "E"),      # D fails against C; E is compared with C
        ((1.2, 1.2, 1.2), (1.1, 1.1, 1.1), "C"),
    ],
)
def test_select_final_prefers_simpler_unless_consistently_better(d, e, expected):
    final, steps = ev.select_final(full(d, e))
    assert final == expected
    assert [(s.candidate, s.reference) for s in steps][0] == ("B", "A")


def test_select_final_keeps_a_when_nothing_beats_it():
    errors = {exp: [1.0] * 3 for exp in ev.EXPERIMENTS}
    errors["A"] = [0.5] * 3
    final, _ = ev.select_final(ev.validate_predictions(make_preds(errors, d_groups=())))
    assert final == "A"


def test_select_final_requires_all_experiments(valid):
    with pytest.raises(ValueError, match="needs experiments"):
        ev.select_final(ev.validate_predictions(valid))


def test_select_final_rejects_d_built_from_other_groups():
    with pytest.raises(ValueError, match="ablation retains"):
        ev.select_final(full((0.8, 0.9, 0.95), (0.5, 0.6, 0.7), groups=("lags", "time")))


# --- Error analysis -------------------------------------------------------


def test_condition_labels_use_development_rows_only():
    df = grid_frame()
    labels, thresholds = ev.condition_labels(df)
    assert labels.index.max() < TEST_START
    changed = df.copy()
    changed.loc[changed.index >= TEST_START, [TARGET, "T", "RH"]] = 1e6
    labels2, thresholds2 = ev.condition_labels(changed)
    assert thresholds2 == thresholds
    pd.testing.assert_frame_equal(labels2, labels)


def test_condition_thresholds_follow_eligible_development_rows():
    df = grid_frame()
    labels, thr = ev.condition_labels(df)
    dev = df[df.index < TEST_START]
    assert thr["high_co"] == pytest.approx(dev[TARGET].quantile(0.9))
    assert labels["co_level"].str.startswith("CO >=").mean() == pytest.approx(0.1, abs=0.01)
    assert labels["T_tertile"].nunique() == 3


def test_lag_history_flags_recent_outage_and_unscorable_hours():
    df = grid_frame()
    outage = pd.Timestamp("2004-09-01 10:00")
    df.loc[outage, INPUT_COLS] = np.nan
    labels, _ = ev.condition_labels(df)
    for h in (1, 3, 24):
        assert labels.loc[outage + pd.Timedelta(hours=h), "lag_history"] == "incomplete"
    assert labels.loc[outage + pd.Timedelta(hours=2), "lag_history"] == "complete"
    assert ev.unscorable_hours(df) == 1


def test_error_by_condition_and_over_time(valid):
    preds = ev.validate_predictions(valid)
    hours = preds["timestamp"].unique()
    labels = pd.DataFrame({"half": np.where(np.arange(len(hours)) % 2, "odd", "even")}, index=hours)
    table = ev.error_by_condition(preds, labels, ["B", "C"])
    assert set(table["experiment"]) == {"B", "C"}
    assert table.loc[table["experiment"] == "C", "mae"].tolist() == pytest.approx([1.0, 1.0])
    bias = table[table["experiment"] == "C"].set_index("level")["bias"]
    assert bias.to_dict() == pytest.approx({"even": -1.0, "odd": 1.0})   # sign alternates by hour
    with pytest.raises(ValueError, match="no condition label"):
        ev.error_by_condition(preds, labels.iloc[1:], ["C"])

    weekly = ev.errors_over_time(preds, ["C"])
    assert set(weekly["fold"]) == {"0", "1", "2"}
    assert weekly["mae"].tolist() == pytest.approx([1.0] * len(weekly))


# --- End to end and isolation ---------------------------------------------


def test_run_stage1_reports_ablation_and_skips_selection(valid, tmp_path):
    text = ev.run(valid, None, tmp_path)
    assert "Stage 2 command" in text and "skipped" in text
    assert not list(tmp_path.glob("*.png"))


def test_run_full_writes_three_figures(tmp_path):
    text = ev.run(full((0.8, 0.9, 0.95), (0.5, 0.6, 0.7)), grid_frame(), tmp_path)
    assert "Final candidate (development evidence only): **E**" in text
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "01_cv_mae_by_experiment.png", "02_weekly_mae.png", "03_error_by_condition.png",
    ]


def test_run_refuses_final_test_rows(valid):
    test_rows = valid.head(1).assign(fold="test", timestamp=TEST_START)
    with pytest.raises(ValueError, match="allow_final_test"):
        ev.run(pd.concat([valid, test_rows], ignore_index=True), None)


def test_evaluation_does_not_import_modeling():
    tree = ast.parse(Path(ev.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("modeling" in (m or "") for m in imported)


REAL_CSV = DEFAULT_RAW_DIR / CSV_NAME


@pytest.mark.skipif(not REAL_CSV.exists(), reason="raw CSV not downloaded")
def test_real_data_conditions_match_contract():
    df = load_raw()
    labels, _ = ev.condition_labels(df)
    assert len(labels) == 7494                    # development grid hours (contract §7)
    assert ev.unscorable_hours(df) == 245          # valid target, no current input
