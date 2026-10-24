"""Evaluation of out-of-fold predictions for experiments A–E (Issue #9).

The only input is the predictions file written by the modeling runner
(Issue #8). This module never imports ``air_quality.modeling``. The file has
one row per experiment and scored hour, with the columns in
``PREDICTION_COLS``. :func:`validate_predictions` rejects anything that would
make experiments incomparable, instead of silently dropping it.

Selection follows the rule pre-registered in Issue #4 and the procedure in
Issue #9 (:func:`compare`, :func:`retained_groups`, :func:`select_final`).
Error analysis uses development rows only. Run
``python -m air_quality.evaluation`` after the runner has written its file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from air_quality.data import (
    CSV_NAME,
    DEFAULT_RAW_DIR,
    PROJECT_ROOT,
    TARGET,
    eligible_mask,
    inputs_available,
    load_raw,
)
from air_quality.features import FEATURE_GROUPS, build_features
from air_quality.split import N_SPLITS, TEST_START, dev_folds, split_dev_test

PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "cv_predictions.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "modeling"

# --- Shared interface with Issue #8 ---------------------------------------

PREDICTION_COLS = ["experiment", "model", "feature_set", "fold", "timestamp", "y_true", "y_pred"]
CANDIDATES = tuple(f"C+{g}" for g in FEATURE_GROUPS)
EXPERIMENTS = ("A", "B", "C", *CANDIDATES, "D", "E")
EXPECTED_MODEL = {"A": "dummy_median", "E": "hgb", **{e: "ridge" for e in ("B", "C", *CANDIDATES, "D")}}
# Feature sets fixed by the experiment design. A uses no features, and D/E
# depend on the ablation outcome, so they are checked in select_final().
EXPECTED_FEATURES = {
    "B": {"S1"},
    "C": {"inputs"},
    **{f"C+{g}": {"inputs", g} for g in FEATURE_GROUPS},
}
CV_FOLDS = tuple(str(i) for i in range(N_SPLITS))
TEST_FOLD = "test"

# --- Pre-registered selection (Issues #4 and #9) --------------------------

# Majority of the primary folds: 2 of 3 under contract v1.1.
MIN_FOLDS_IMPROVED = N_SPLITS // 2 + 1
# Order of increasing complexity. A is the baseline reference; each later
# experiment replaces the current choice only if it passes the rule against it.
SELECTION_ORDER = ("A", "B", "C", "D", "E")

# --- Error-analysis definitions (fixed before any real result) ------------

HIGH_CO_QUANTILE = 0.90       # of eligible development CO(GT)
CONDITION_QUANTILES = (1 / 3, 2 / 3)  # tertiles of eligible development T and RH


def _fail(message: str) -> None:
    raise ValueError(f"Invalid predictions: {message}")


def _groups_of(feature_set: str) -> set[str]:
    return set(feature_set.split("+"))


def validate_predictions(preds: pd.DataFrame, *, allow_final_test: bool = False) -> pd.DataFrame:
    """Check a predictions frame against the shared interface; return a clean copy.

    Raises ``ValueError`` on any of: missing or extra columns; missing or
    non-finite values; unknown experiments or folds; a model or feature set
    that does not match the experiment design or changes within an
    experiment; a duplicated (experiment, hour); a CV row outside its fold's
    validation window or at or after ``TEST_START``; an experiment missing a
    fold; different scored hours between experiments in the same fold; or
    different ``y_true`` for the same hour.

    Final-test rows (``fold == "test"``) are rejected unless
    ``allow_final_test`` is true. That option exists only for the one final
    evaluation in the Issue #4 integration.
    """
    missing = [c for c in PREDICTION_COLS if c not in preds.columns]
    extra = [c for c in preds.columns if c not in PREDICTION_COLS]
    if missing or extra:
        _fail(f"missing columns {missing}, unexpected columns {extra}")
    if preds.empty:
        _fail("no rows")

    out = preds[PREDICTION_COLS].copy()
    nan_counts = out.isna().sum()
    if nan_counts.any():
        _fail(f"missing values per column {nan_counts[nan_counts > 0].to_dict()}")
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        _fail("timestamp is not a datetime column")
    for col in ("y_true", "y_pred"):
        if not pd.api.types.is_numeric_dtype(out[col]) or not np.isfinite(out[col]).all():
            _fail(f"{col} must be finite numbers")
    for col in ("experiment", "model", "feature_set", "fold"):
        out[col] = out[col].astype(str)

    unknown = sorted(set(out["experiment"]) - set(EXPERIMENTS))
    if unknown:
        _fail(f"unknown experiments {unknown}")
    allowed_folds = (*CV_FOLDS, TEST_FOLD) if allow_final_test else CV_FOLDS
    bad_folds = sorted(set(out["fold"]) - set(allowed_folds))
    if bad_folds:
        hint = " (final-test rows need allow_final_test=True)" if TEST_FOLD in bad_folds else ""
        _fail(f"unexpected folds {bad_folds}{hint}")

    for col in ("model", "feature_set"):
        varying = out.groupby("experiment")[col].nunique()
        if (varying > 1).any():
            _fail(f"{col} changes within experiments {sorted(varying[varying > 1].index)}")
    design = out.groupby("experiment")[["model", "feature_set"]].first()
    for exp, row in design.iterrows():
        if row["model"] != EXPECTED_MODEL[exp]:
            _fail(f"experiment {exp} has model {row['model']!r}, expected {EXPECTED_MODEL[exp]!r}")
        if exp in EXPECTED_FEATURES and _groups_of(row["feature_set"]) != EXPECTED_FEATURES[exp]:
            _fail(f"experiment {exp} has feature_set {row['feature_set']!r}")

    dup = out.duplicated(["experiment", "timestamp"], keep=False)
    if dup.any():
        _fail(f"{int(dup.sum())} rows repeat an (experiment, timestamp) pair")

    is_cv = out["fold"] != TEST_FOLD
    if (out.loc[is_cv, "timestamp"] >= TEST_START).any():
        _fail(f"CV rows at or after TEST_START ({TEST_START})")
    if (out.loc[~is_cv, "timestamp"] < TEST_START).any():
        _fail("final-test rows before TEST_START")
    for label, fold in zip(CV_FOLDS, dev_folds()):
        ts = out.loc[out["fold"] == label, "timestamp"]
        outside = (ts < fold.val_start) | (ts >= fold.val_end)
        if outside.any():
            _fail(f"{int(outside.sum())} rows of fold {label} lie outside its validation window")

    cv_experiments = set(out.loc[is_cv, "experiment"])
    for label in CV_FOLDS:
        present = set(out.loc[out["fold"] == label, "experiment"])
        if present != cv_experiments:
            _fail(f"fold {label} lacks experiments {sorted(cv_experiments - present)}")
    for label, rows in out.groupby("fold"):
        hours = rows.groupby("experiment")["timestamp"].apply(frozenset)
        if hours.nunique() > 1:
            counts = rows.groupby("experiment").size().to_dict()
            _fail(f"fold {label} scores different hours per experiment (row counts {counts})")

    if (out.groupby("timestamp")["y_true"].nunique() > 1).any():
        _fail("y_true differs between experiments for the same hour")

    return out.sort_values(["experiment", "timestamp"]).reset_index(drop=True)


def load_predictions(path: Path | None = None, *, allow_final_test: bool = False) -> pd.DataFrame:
    """Read the shared predictions CSV and validate it (see :func:`validate_predictions`)."""
    path = Path(path) if path is not None else PREDICTIONS_PATH
    raw = pd.read_csv(
        path,
        dtype={"experiment": str, "model": str, "feature_set": str, "fold": str},
        parse_dates=["timestamp"],
    )
    return validate_predictions(raw, allow_final_test=allow_final_test)


# --- Metrics --------------------------------------------------------------


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE (primary), RMSE and R²."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _cv_rows(preds: pd.DataFrame) -> pd.DataFrame:
    return preds[preds["fold"] != TEST_FOLD]


def _ordered(experiments) -> list[str]:
    return [e for e in EXPERIMENTS if e in set(experiments)]


def fold_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per (experiment, CV fold): row count, MAE, RMSE and R²."""
    rows = []
    for (exp, fold), g in _cv_rows(preds).groupby(["experiment", "fold"]):
        rows.append({
            "experiment": exp,
            "model": g["model"].iat[0],
            "feature_set": g["feature_set"].iat[0],
            "fold": fold,
            "n": len(g),
            **regression_metrics(g["y_true"], g["y_pred"]),
        })
    out = pd.DataFrame(rows)
    out["experiment"] = pd.Categorical(out["experiment"], _ordered(out["experiment"]), ordered=True)
    return out.sort_values(["experiment", "fold"]).reset_index(drop=True)


def cv_summary(preds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (ddof=1) across CV folds, plus per-fold MAE."""
    per_fold = fold_metrics(preds)
    grouped = per_fold.groupby("experiment", observed=True)
    out = grouped[["model", "feature_set"]].first()
    out["n_folds"] = grouped.size()
    for metric in ("mae", "rmse", "r2"):
        out[f"{metric}_mean"] = grouped[metric].mean()
        out[f"{metric}_std"] = grouped[metric].std(ddof=1)
    fold_mae = per_fold.pivot(index="experiment", columns="fold", values="mae")
    for fold in fold_mae.columns:
        out[f"mae_fold_{fold}"] = fold_mae[fold]
    out.index = out.index.astype(str)
    return out


# --- Selection ------------------------------------------------------------


class Comparison(NamedTuple):
    candidate: str
    reference: str
    candidate_mae: float          # mean CV MAE
    reference_mae: float
    candidate_fold_mae: tuple[float, ...]
    reference_fold_mae: tuple[float, ...]
    folds_improved: int
    passes: bool


def compare(preds: pd.DataFrame, candidate: str, reference: str) -> Comparison:
    """Apply the pre-registered rule: lower mean CV MAE and lower MAE in a majority of folds.

    Ties count as no improvement. ``preds`` must come from
    :func:`validate_predictions`, which guarantees that both experiments are
    scored on identical hours in every fold.
    """
    per_fold = fold_metrics(preds).astype({"experiment": str})
    mae = per_fold.pivot(index="fold", columns="experiment", values="mae")
    for exp in (candidate, reference):
        if exp not in mae.columns:
            raise KeyError(f"Experiment {exp} is not in the predictions")
    cand, ref = mae[candidate], mae[reference]
    improved = int((cand < ref).sum())
    passes = bool(cand.mean() < ref.mean() and improved >= MIN_FOLDS_IMPROVED)
    return Comparison(
        candidate, reference, float(cand.mean()), float(ref.mean()),
        tuple(cand.round(6)), tuple(ref.round(6)), improved, passes,
    )


def ablation_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Each feature group added alone to C, compared with C (Issue #9, step 1)."""
    rows = []
    for group in FEATURE_GROUPS:
        c = compare(preds, f"C+{group}", "C")
        rows.append({
            "group": group,
            "experiment": c.candidate,
            "mae_mean": c.candidate_mae,
            "mae_mean_C": c.reference_mae,
            "delta_mae": c.candidate_mae - c.reference_mae,
            "folds_improved": f"{c.folds_improved}/{len(c.candidate_fold_mae)}",
            "retained": c.passes,
        })
    return pd.DataFrame(rows)


def retained_groups(preds: pd.DataFrame) -> list[str]:
    """Feature groups that pass the rule against C, in ``FEATURE_GROUPS`` order."""
    table = ablation_table(preds)
    return table.loc[table["retained"], "group"].tolist()


def d_feature_set(groups: Sequence[str]) -> str:
    """Feature-set label of D and E for the given retained groups."""
    return "+".join(["inputs", *groups])


def select_final(preds: pd.DataFrame) -> tuple[str, list[Comparison]]:
    """Pre-registered final-candidate procedure on development CV evidence only.

    Walks ``SELECTION_ORDER`` (A → B → C → D → E). Each experiment replaces
    the current choice only if it passes :func:`compare` against it. D and E
    must use exactly C's inputs plus the groups retained by the ablation.
    Returns the chosen experiment and every comparison made.
    """
    present = set(preds["experiment"])
    missing = [e for e in (*SELECTION_ORDER, *CANDIDATES) if e not in present]
    if missing:
        raise ValueError(f"Final selection needs experiments {missing}")
    expected = _groups_of(d_feature_set(retained_groups(preds)))
    for exp in ("D", "E"):
        fs = preds.loc[preds["experiment"] == exp, "feature_set"].iat[0]
        if _groups_of(fs) != expected:
            raise ValueError(
                f"{exp} uses feature_set {fs!r}, but the ablation retains {sorted(expected)}"
            )

    current, steps = SELECTION_ORDER[0], []
    for exp in SELECTION_ORDER[1:]:
        step = compare(preds, exp, current)
        steps.append(step)
        if step.passes:
            current = exp
    return current, steps


# --- Error analysis (development period only) -----------------------------


def condition_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Condition labels for every development hour, and the thresholds used.

    ``df`` is the complete hourly grid from :func:`load_raw`. Only
    development rows are labelled. Thresholds come from eligible development
    rows: the ``HIGH_CO_QUANTILE`` of ``CO(GT)`` and the tertiles of ``T``
    and ``RH``. ``lag_history`` is ``incomplete`` when any sensor lag at
    1, 3 or 24 h is missing, i.e. an input outage in that recent past.
    """
    dev, _ = split_dev_test(df)
    eligible = dev[eligible_mask(dev)]
    thresholds = {"high_co": float(eligible[TARGET].quantile(HIGH_CO_QUANTILE))}

    def ordered(values, categories):
        return pd.Categorical(values, categories=categories, ordered=True)

    labels = pd.DataFrame(index=dev.index)
    high = f"CO >= {thresholds['high_co']:.2f}"
    labels["co_level"] = ordered(np.where(dev[TARGET] >= thresholds["high_co"], high, "other"),
                                 ["other", high])
    for col in ("T", "RH"):
        lo, hi = eligible[col].quantile(CONDITION_QUANTILES)
        thresholds[f"{col}_low"], thresholds[f"{col}_high"] = float(lo), float(hi)
        labels[f"{col}_tertile"] = pd.cut(
            dev[col], [-np.inf, lo, hi, np.inf], right=False,
            labels=[f"< {lo:.1f}", f"{lo:.1f}–{hi:.1f}", f">= {hi:.1f}"],
        )
    # Lags are built on the full grid (contract §10) and then restricted to dev.
    lags = build_features(df, groups="lags").loc[dev.index]
    labels["lag_history"] = ordered(np.where(lags.isna().any(axis=1), "incomplete", "complete"),
                                    ["complete", "incomplete"])
    return labels, thresholds


def unscorable_hours(df: pd.DataFrame) -> int:
    """Development hours with a valid target but no current input (cannot be scored)."""
    dev, _ = split_dev_test(df)
    return int((dev[TARGET].notna() & ~inputs_available(dev)).sum())


def error_by_condition(
    preds: pd.DataFrame, labels: pd.DataFrame, experiments: Sequence[str]
) -> pd.DataFrame:
    """MAE, RMSE and bias (mean of prediction minus truth) per condition level."""
    cv = _cv_rows(preds)
    cv = cv[cv["experiment"].isin(experiments)]
    unlabelled = ~cv["timestamp"].isin(labels.index)
    if unlabelled.any():
        raise ValueError(f"{int(unlabelled.sum())} scored hours have no condition label")
    joined = cv.join(labels, on="timestamp")
    rows = []
    for condition in labels.columns:
        for (level, exp), g in joined.groupby([condition, "experiment"], observed=True):
            err = g["y_pred"] - g["y_true"]
            rows.append({
                "condition": condition,
                "level": level,
                "experiment": exp,
                "n": len(g),
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err ** 2).mean())),
                "bias": float(err.mean()),
            })
    return pd.DataFrame(rows)


def errors_over_time(
    preds: pd.DataFrame, experiments: Sequence[str], freq: str = "W"
) -> pd.DataFrame:
    """MAE per period (weekly by default), kept separate per CV fold.

    Weeks are split at fold boundaries, so each value comes from one fitted
    model. This is descriptive: season and sensor drift are confounded.
    """
    cv = _cv_rows(preds)
    cv = cv[cv["experiment"].isin(experiments)].copy()
    cv["period_start"] = cv["timestamp"].dt.to_period(freq).dt.start_time
    cv["abs_error"] = (cv["y_pred"] - cv["y_true"]).abs()
    out = (
        cv.groupby(["experiment", "fold", "period_start"])
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"), first_hour=("timestamp", "min"))
        .reset_index()
    )
    return out.sort_values(["experiment", "first_hour"]).reset_index(drop=True)


# --- Figures --------------------------------------------------------------

_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"
_SERIES = ("#2a78d6", "#eb6834")   # categorical slots 1-2: final candidate, reference
_CONDITION_TITLES = {
    "co_level": "CO(GT) level (mg/m³)",
    "T_tertile": "Temperature T (°C)",
    "RH_tertile": "Relative humidity RH (%)",
    "lag_history": "Sensor lag history (1, 3, 24 h)",
}


def _axes_style(ax) -> None:
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_cv_mae(preds: pd.DataFrame, path: Path, experiments: Sequence[str] = SELECTION_ORDER) -> Path:
    """Per-fold and mean CV MAE for the main experiments."""
    plt = _pyplot()
    per_fold = fold_metrics(preds).astype({"experiment": str})
    exps = [e for e in experiments if e in set(per_fold["experiment"])]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, exp in enumerate(exps):
        vals = per_fold.loc[per_fold["experiment"] == exp, "mae"].to_numpy()
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=36,
                   color=_MUTED, alpha=0.6, label="Fold MAE" if i == 0 else None, zorder=2)
        ax.hlines(vals.mean(), i - 0.28, i + 0.28, color=_SERIES[0], linewidth=2.5,
                  label="Mean across folds" if i == 0 else None, zorder=3)
        ax.annotate(f"{vals.mean():.3f}", (i + 0.3, vals.mean()), va="center", fontsize=8, color=_INK)
    ax.set_xticks(range(len(exps)), exps)
    ax.set_xlim(-0.6, len(exps) - 0.2)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MAE (mg/m³)", color=_INK)
    ax.set_title("Cross-validated MAE by experiment (3 primary folds)", color=_INK, loc="left")
    _axes_style(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_errors_over_time(weekly: pd.DataFrame, experiments: Sequence[str], path: Path) -> Path:
    """Weekly MAE across the validation windows, one segment per fold."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for color, exp in zip(_SERIES, experiments):
        for j, (_, seg) in enumerate(weekly[weekly["experiment"] == exp].groupby("fold")):
            ax.plot(seg["first_hour"], seg["mae"], color=color, linewidth=2, marker="o",
                    markersize=4, label=exp if j == 0 else None)
    for fold in dev_folds()[1:]:
        ax.axvline(fold.val_start, color=_MUTED, linestyle=":", linewidth=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Weekly MAE (mg/m³)", color=_INK)
    ax.set_title("Validation error over time (dotted lines: fold boundaries)", color=_INK, loc="left")
    _axes_style(ax)
    ax.legend(frameon=False, title="Experiment")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_error_by_condition(table: pd.DataFrame, experiments: Sequence[str], path: Path) -> Path:
    """MAE per condition level, one panel per condition."""
    plt = _pyplot()
    conditions = list(dict.fromkeys(table["condition"]))
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.2 * len(conditions), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / len(experiments)
    for ax, condition in zip(axes, conditions):
        sub = table[table["condition"] == condition]
        levels = list(dict.fromkeys(sub["level"]))
        for k, (color, exp) in enumerate(zip(_SERIES, experiments)):
            mae = sub[sub["experiment"] == exp].set_index("level")["mae"].reindex(levels)
            ax.bar(np.arange(len(levels)) + (k - (len(experiments) - 1) / 2) * width, mae,
                   width=width, color=color, edgecolor="white", linewidth=2, label=exp)
        ax.set_xticks(range(len(levels)), levels, rotation=20, ha="right", fontsize=8)
        ax.set_title(_CONDITION_TITLES.get(condition, condition), color=_INK, fontsize=10)
        _axes_style(ax)
    axes[0].set_ylabel("MAE (mg/m³)", color=_INK)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, title="Experiment", frameon=False, ncol=len(names),
               loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# --- Report output --------------------------------------------------------


def to_markdown(table: pd.DataFrame, floatfmt: str = ".3f", index: bool = False) -> str:
    """Render a small table as GitHub Markdown (no extra dependency)."""
    t = table.reset_index() if index else table
    def cell(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def run(preds: pd.DataFrame, df: pd.DataFrame | None, figures_dir: Path = FIGURES_DIR) -> str:
    """Produce the evaluation tables (Markdown) and figures for validated CV predictions.

    With only A, B, C and the five candidates present (runner stage 1), it
    reports the ablation and the groups to pass to ``--d-groups``. Once D and
    E are present, it also applies :func:`select_final` and, if ``df`` is
    given, the development error analysis and figures.
    """
    preds = validate_predictions(preds)   # development CV rows only
    present = set(preds["experiment"])
    parts = ["## CV summary (primary 3-fold protocol, contract v1.1)",
             to_markdown(cv_summary(preds), index=True)]

    if {"C", *CANDIDATES} <= present:
        groups = retained_groups(preds)
        parts += ["## Feature-group ablation (each group vs C)", to_markdown(ablation_table(preds)),
                  f"Retained groups: {', '.join(groups) if groups else 'none'}",
                  f"Stage 2 command: `python -m air_quality.modeling --d-groups {','.join(groups)}`"]
    if not ({*SELECTION_ORDER, *CANDIDATES} <= present):
        parts.append("D and/or E are not present yet: final selection and error analysis skipped.")
        return "\n\n".join(parts)

    final, steps = select_final(preds)
    parts += ["## Final-candidate procedure (A → B → C → D → E)",
              to_markdown(pd.DataFrame(steps).drop(columns=["candidate_fold_mae", "reference_fold_mae"])),
              f"Final candidate (development evidence only): **{final}**"]
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_cv_mae(preds, figures_dir / "01_cv_mae_by_experiment.png")
    if df is None:
        return "\n\n".join(parts)

    shown = list(dict.fromkeys([final, "B"]))
    labels, thresholds = condition_labels(df)
    by_condition = error_by_condition(preds, labels, shown)
    weekly = errors_over_time(preds, shown)
    plot_errors_over_time(weekly, shown, figures_dir / "02_weekly_mae.png")
    plot_error_by_condition(by_condition, shown, figures_dir / "03_error_by_condition.png")
    parts += ["## Error analysis (development CV predictions)",
              "Thresholds: " + ", ".join(f"{k} = {v:.3f}" for k, v in thresholds.items()),
              f"Development hours with a valid target but no current input (not scored): "
              f"{unscorable_hours(df)}",
              to_markdown(by_condition)]
    return "\n\n".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate CV predictions (Issue #9).")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR / CSV_NAME,
                        help="raw CSV used for condition labels (development rows only)")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--out", type=Path, help="also write the Markdown tables to this file")
    args = parser.parse_args(argv)

    preds = load_predictions(args.predictions)
    text = run(preds, load_raw(args.raw), args.figures_dir)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
