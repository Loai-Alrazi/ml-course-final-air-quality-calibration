# Model Results: Experiments A–E

**Issue:** #9 (part of #4) — Evaluation, Error Analysis & Results Report
**Data contract:** v1.1 (`docs/data_contract.md`)
**Status:** Pre-registration only. **No experiment has been run yet, and this file contains no results.** The result sections below are filled from the predictions file produced by the #8 runner. The final-test section is filled once, in the #4 integration.

This file is committed before any real number exists, so that the selection procedure and the error-analysis definitions are fixed in advance and visible in the git history.

---

## 1. Protocol

- **Task:** current-hour estimation of `CO(GT)` (mg/m³) from low-cost inputs (`PROJECT_SPEC.md` §3).
- **Validation:** the three primary expanding-window folds of contract v1.1 (`dev_folds()`), with validation windows starting 2004-08-14, 2004-10-05 and 2004-11-26. Every experiment is scored on exactly the same eligible development hours in each fold. `evaluation.validate_predictions` enforces this and stops on any difference.
- **Learned preprocessing:** fitted inside each training fold only (#8).
- **Final test:** from `TEST_START` (2005-01-17). It is not used in this file before the integration. `evaluation.py` rejects final-test rows unless that is explicitly allowed.
- **Metrics:** MAE (primary), plus RMSE and R² (secondary). Each is reported as the mean and standard deviation (ddof = 1) across the three folds, with per-fold MAE. The mean CV MAE is the mean of the per-fold MAEs.

## 2. Experiments

| Experiment | Model | Features |
|---|---|---|
| A | `DummyRegressor(strategy="median")` | none |
| B | Ridge | `PT08.S1(CO)` |
| C | Ridge | the five sensors + `T`, `RH`, `AH` |
| `C+<group>` | Ridge | C + one group from #3 (`time`, `lags`, `rolling`, `env_interaction`, `missing_ind`) |
| D | Ridge | C + the groups retained by the ablation |
| E | `HistGradientBoostingRegressor` | the same features as D |

Fixed model settings are recorded by #8 before the first run.

## 3. Pre-registered selection

**Rule (Issue #4).** A candidate beats a reference only if it has a lower mean CV MAE **and** a lower MAE in at least 2 of the 3 folds. Ties count as no improvement.

**Procedure (Issue #9), implemented in `evaluation.py`:**

1. **Ablation.** Each group is tested alone against C (`C+<group>` vs `C`). D = C plus every group that passes. If no group passes, D equals C, reported as "no group retained". `select_final` checks that D and E were run with exactly these groups.
2. **E** uses D's feature set.
3. **Final candidate.** The experiments are walked in order of complexity, A → B → C → D → E. A is the baseline reference. Each later experiment replaces the current choice only if it passes the rule against that choice. If nothing beats A, this report says so.

The procedure yields the development-period candidate. The Project Lead freezes it before the final test.

## 4. Error-analysis definitions

The error analysis uses development CV predictions only. It is applied to the final candidate, with B as the reference. Thresholds are computed from eligible development rows.

| Condition | Definition |
|---|---|
| Errors over time | Weekly MAE within each validation window; weeks split at fold boundaries |
| High CO | `CO(GT)` ≥ the 90th percentile of eligible development `CO(GT)` |
| Temperature / humidity | Tertiles of `T` and `RH` |
| Missing sensor readings | `lag_history = incomplete` when any sensor lag at 1, 3 or 24 h is missing, i.e. an input outage in that recent past. Separately, the development hours with a valid target but no current input are counted: they cannot be scored (245, contract §7) |

Changes over time are described, not interpreted as sensor drift: season and drift are confounded in this data (contract §8, `docs/data_audit.md` §7.2).

## 5. Reproduction

```bash
python -m air_quality.modeling                          # stage 1 (#8): A, B, C and the five candidates
python -m air_quality.evaluation                        # ablation table and retained groups
python -m air_quality.modeling --d-groups <retained>    # stage 2 (#8): D and E
python -m air_quality.evaluation --out outputs/evaluation_tables.md   # tables + figures in reports/figures/modeling/
```

The predictions file (`outputs/cv_predictions.csv`) is git-ignored. The commit and SHA-256 of the file used for this report will be recorded here.

## 6. Results (pending #8 predictions)

### 6.1 A–E summary

_Pending._

### 6.2 Feature-group ablation

_Pending._

### 6.3 Final-candidate procedure

_Pending._

### 6.4 Error analysis

_Pending._

## 7. Final test (#4 integration only)

_Not run. The Project Lead runs it once, after the decisions above are frozen._

## 8. Limitations

- Three validation folds give only three per-fold estimates, so their spread is a rough indication of variability.
- Validation covers mid-August 2004 to mid-January 2005 only, with no spring window (contract §9).
- Long target gaps in August–October 2004 reduce the scored hours in fold 0.
- Hours without current inputs cannot be estimated and are excluded from scoring.
- Season and sensor drift cannot be separated with this data.
