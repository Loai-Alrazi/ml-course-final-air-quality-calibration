"""Chronological development/final-test split and time-aware CV folds.

All boundaries are timestamps on the complete hourly grid, fixed before any
row is filtered by target or input availability. Filtering is applied
afterwards, inside each period or fold. See ``docs/data_contract.md``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from air_quality.data import eligible_mask, inputs_available

# First hour of the verified file's grid. Folds are derived from this constant,
# never from a filtered index, so every experiment gets identical boundaries.
DATA_START = pd.Timestamp("2004-03-10 18:00:00")

# Final-test boundary: nearest midnight to 80% of the timeline (2005-01-16 14:48).
# Rows at or after this timestamp are the final test and are never used for
# feature, model, or preprocessing decisions.
TEST_START = pd.Timestamp("2005-01-17 00:00:00")

N_SPLITS = 5
# No gap between training and validation windows. Features use only current
# and past low-cost inputs, never the target, so validation rows cannot leak
# label information into training; the final test also follows the
# development period without a gap. The parameter stays for sensitivity checks.
GAP_HOURS = 0


class Fold(NamedTuple):
    """Half-open intervals: train = [train_start, train_end), val = [val_start, val_end)."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def split_dev_test(df: pd.DataFrame, test_start: pd.Timestamp = TEST_START):
    """Split a timestamp-indexed frame into (development, final_test) by time only."""
    return df[df.index < test_start], df[df.index >= test_start]


def dev_folds(
    *,
    data_start: pd.Timestamp = DATA_START,
    n_splits: int = N_SPLITS,
    gap_hours: int = GAP_HOURS,
    dev_end: pd.Timestamp = TEST_START,
) -> list[Fold]:
    """Expanding-window folds over the development period's hourly grid.

    Like ``TimeSeriesSplit``, the grid is cut into ``n_splits + 1`` equal
    blocks: training starts with the first block and each later block is one
    validation window. Validation boundaries are rounded to the nearest
    midnight so they are easy to document; the last window ends at ``dev_end``.

    Call ``dev_folds()`` with no arguments for the shared protocol. The
    keyword-only overrides exist for unit tests on other grids and for
    documented sensitivity checks (e.g. ``gap_hours=24``). Never pass the
    first timestamp of a filtered or feature-engineered frame as
    ``data_start``: dropping leading rows would silently move every boundary.
    """
    hours = pd.date_range(data_start, dev_end, freq="h", inclusive="left")
    val_size = len(hours) // (n_splits + 1)
    first = len(hours) - n_splits * val_size
    cuts = [hours[first + k * val_size].round("D") for k in range(n_splits)] + [dev_end]
    if any(b <= a for a, b in zip([data_start, *cuts[:-1]], cuts)):
        raise ValueError("Fold boundaries are not strictly increasing")

    gap = pd.Timedelta(hours=gap_hours)
    folds = []
    for val_start, val_end in zip(cuts[:-1], cuts[1:]):
        train_end = val_start - gap
        if train_end <= data_start:
            raise ValueError("Gap leaves an empty training window")
        folds.append(Fold(data_start, train_end, val_start, val_end))
    return folds


def cv_indices(index: pd.DatetimeIndex, folds: list[Fold]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Positional (train, val) indices of ``index`` for each fold.

    ``index`` may already be filtered (e.g. to eligible development rows).
    Compute it from the exact index of the ``X``/``y`` passed to scikit-learn,
    after all row filtering, and pass the result as ``cv=`` to
    ``cross_validate``. ``cross_val_predict`` does not accept it, because the
    first block is training-only (expanding windows are not a partition).
    """
    splits = []
    for i, fold in enumerate(folds):
        train = np.flatnonzero((index >= fold.train_start) & (index < fold.train_end))
        val = np.flatnonzero((index >= fold.val_start) & (index < fold.val_end))
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"Fold {i} has an empty training or validation set")
        splits.append((train, val))
    return splits


def fold_summary(dev: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Dates, grid hours, input availability, and eligible-row counts per fold.

    ``*_inputs_available`` counts hours with at least one low-cost input
    (what is known at prediction time); ``*_eligible`` additionally requires a
    valid reference target (what can be scored). ``val_eligible_share`` is
    eligible validation rows divided by validation grid hours.
    """
    eligible = eligible_mask(dev)
    has_inputs = inputs_available(dev)
    rows = []
    for i, f in enumerate(folds):
        in_train = (dev.index >= f.train_start) & (dev.index < f.train_end)
        in_val = (dev.index >= f.val_start) & (dev.index < f.val_end)
        rows.append({
            "fold": i,
            "train_start": f.train_start,
            "train_end": f.train_end,
            "val_start": f.val_start,
            "val_end": f.val_end,
            "train_hours": int(in_train.sum()),
            "val_hours": int(in_val.sum()),
            "val_inputs_available": int((in_val & has_inputs).sum()),
            "train_eligible": int((in_train & eligible).sum()),
            "val_eligible": int((in_val & eligible).sum()),
        })
    out = pd.DataFrame(rows)
    out["val_eligible_share"] = (out["val_eligible"] / out["val_hours"]).round(3)
    return out
