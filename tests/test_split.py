import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_predict, cross_validate

from air_quality.data import CSV_NAME, DEFAULT_RAW_DIR, INPUT_COLS, TARGET, eligible_mask, load_raw
from air_quality.split import (
    DATA_START,
    TEST_START,
    cv_indices,
    dev_folds,
    fold_summary,
    split_dev_test,
)


def hourly_frame(start=DATA_START, end="2005-04-04 14:00"):
    index = pd.date_range(start, end, freq="h", name="timestamp")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(len(index), len(INPUT_COLS))), index=index, columns=INPUT_COLS)
    df[TARGET] = rng.uniform(0.1, 5, len(index))
    return df


def test_dev_test_split_is_chronological_and_disjoint():
    df = hourly_frame()
    dev, test = split_dev_test(df)
    assert len(dev) + len(test) == len(df)
    assert dev.index.max() < TEST_START <= test.index.min()
    assert dev.index.intersection(test.index).empty


def test_split_boundary_ignores_target_filtering():
    df = hourly_frame()
    df.loc["2005-01-10":"2005-01-25", TARGET] = np.nan   # missing labels around the boundary
    dev, test = split_dev_test(df)
    assert test.index.min() == TEST_START
    assert dev.index.max() == TEST_START - pd.Timedelta(hours=1)


def test_folds_are_expanding_ordered_and_inside_dev():
    folds = dev_folds(data_start=DATA_START)
    assert len(folds) == 5
    for prev, cur in zip(folds, folds[1:]):
        assert cur.val_start == prev.val_end            # contiguous, non-overlapping windows
        assert cur.train_end > prev.train_end           # expanding training window
    for f in folds:
        assert f.train_start == DATA_START
        assert f.train_end <= f.val_start < f.val_end <= TEST_START
        assert f.val_start == f.val_start.normalize()   # midnight boundaries
    assert folds[-1].val_end == TEST_START


def test_gap_separates_train_and_validation():
    for f in dev_folds(gap_hours=24):
        assert f.val_start - f.train_end == pd.Timedelta(hours=24)


def test_cv_indices_respect_fold_bounds_after_filtering():
    df = hourly_frame()
    df.iloc[::7, df.columns.get_loc(TARGET)] = np.nan
    dev, _ = split_dev_test(df)
    usable = dev[eligible_mask(dev)]
    folds = dev_folds(data_start=DATA_START)
    for (train, val), f in zip(cv_indices(usable.index, folds), folds):
        assert usable.index[train].max() < f.train_end
        assert usable.index[val].min() >= f.val_start
        assert usable.index[val].max() < f.val_end
        assert usable.index[train].max() < usable.index[val].min()


def test_folds_never_reach_the_final_test():
    df = hourly_frame()
    folds = dev_folds(data_start=DATA_START)
    for train, val in cv_indices(df.index, folds):
        assert df.index[np.concatenate([train, val])].max() < TEST_START


def test_default_folds_do_not_depend_on_filtered_data():
    # Boundaries come from the DATA_START constant, so filtering rows away
    # (even the first ones) can never move them.
    assert dev_folds() == dev_folds(data_start=DATA_START)
    df = hourly_frame()
    df.iloc[:30, df.columns.get_loc(TARGET)] = np.nan   # e.g. leading rows lost to lag features
    dev, _ = split_dev_test(df)
    usable = dev[eligible_mask(dev)]
    assert usable.index[0] != DATA_START
    # The hazard the keyword-only signature guards against:
    assert dev_folds(data_start=usable.index[0]) != dev_folds()
    with pytest.raises(TypeError):
        dev_folds(usable.index[0])   # positional start is rejected


def test_cv_indices_work_with_sklearn_cross_validate():
    df = hourly_frame()
    df.iloc[::5, df.columns.get_loc(TARGET)] = np.nan
    dev, _ = split_dev_test(df)
    usable = dev[eligible_mask(dev)]
    cv = cv_indices(usable.index, dev_folds())
    result = cross_validate(DummyRegressor(), usable[INPUT_COLS], usable[TARGET],
                            cv=cv, return_indices=True)
    assert len(result["test_score"]) == len(cv)
    for (train, val), used_train, used_val in zip(cv, result["indices"]["train"], result["indices"]["test"]):
        assert np.array_equal(train, used_train) and np.array_equal(val, used_val)


def test_cross_val_predict_is_not_supported():
    # Expanding windows leave the first block training-only, so they are not a partition.
    df = hourly_frame()
    dev, _ = split_dev_test(df)
    with pytest.raises(ValueError, match="partition"):
        cross_val_predict(DummyRegressor(), dev[INPUT_COLS], dev[TARGET], cv=cv_indices(dev.index, dev_folds()))


def test_fold_summary_counts():
    df = hourly_frame()
    dev, _ = split_dev_test(df)
    summary = fold_summary(dev, dev_folds(data_start=DATA_START))
    assert (summary["val_eligible"] == summary["val_hours"]).all()   # synthetic data: all eligible
    assert summary["train_hours"].is_monotonic_increasing


REAL_CSV = DEFAULT_RAW_DIR / CSV_NAME


@pytest.mark.skipif(not REAL_CSV.exists(), reason="raw CSV not downloaded")
def test_real_split_and_folds_match_documented_values():
    df = load_raw(REAL_CSV)
    assert df.index[0] == DATA_START
    dev, test = split_dev_test(df)
    assert (len(dev), len(test)) == (7494, 1863)
    assert (int(eligible_mask(dev).sum()), int(eligible_mask(test).sum())) == (5657, 1687)

    folds = dev_folds()
    assert [str(f.val_start.date()) for f in folds] == [
        "2004-05-02", "2004-06-23", "2004-08-14", "2004-10-05", "2004-11-26"]
    summary = fold_summary(dev, folds)
    assert summary["val_eligible"].tolist() == [987, 976, 771, 936, 1004]
    assert summary["val_inputs_available"].tolist() == [1196, 1243, 1184, 1248, 1116]
