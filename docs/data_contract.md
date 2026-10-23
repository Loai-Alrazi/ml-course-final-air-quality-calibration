# Data Contract & Baseline Evaluation Protocol

**Version 1.0** (Issue #1)

All numbers below were measured from the downloaded file with the code in `src/air_quality/` unless marked as published metadata. The tests in `tests/test_data.py` and `tests/test_split.py` pin the key values.

## 0. How to use and change this contract

This document fixes what every experiment must share so that results can be compared. It does not restrict how team members design features, models or analyses within those rules.

**Fixed scientific constraints.** These follow from the problem definition and `PROJECT_SPEC.md`. They change only if the specification changes.

- The target is `CO(GT)`. `FORBIDDEN_COLS` and current or past `CO(GT)` are never model inputs.
- Features use only information available at prediction time: current and past low-cost inputs, and the timestamp.
- The final test (from `TEST_START`) stays isolated. It is never used for feature, model or preprocessing choices, and it is evaluated once, after decisions are frozen.
- Boundaries are fixed in time before any row filtering. Learned preprocessing is fitted on training rows only.
- All experiments that are compared with each other use the same protocol version: the same folds and the same eligibility rule.

**Current baseline decisions.** These are revisable with justification: CSV as the reference file; the `TEST_START` value; the eligibility rule; the number and placement of CV folds; `GAP_HOURS`; naive handling of local timestamps; loader details.

**Free within the protocol (no approval needed):** any feature-engineering, EDA or modeling experiment that runs on development data, through `dev_folds()` and `cv_indices()`, without breaking a fixed constraint. Report every such experiment, including unsuccessful ones.

**Changing a baseline decision or a shared interface:**

1. Open a pull request that changes `data.py` or `split.py` (owned by Issue #1). State the reason and the expected effect on comparisons.
2. Update the affected tests and this document, and bump the version with a one-line entry in the change log below.
3. The Project Lead reviews it. If the change alters folds or eligibility, comparisons already run under the old version are re-run before they are compared with new ones.

| Version | Change |
|---|---|
| 1.0 | Initial contract: verified source, loader, eligibility rule, `TEST_START`, CV folds, `GAP_HOURS = 0` |

## 1. Source and reproducibility

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository — [Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) |
| DOI | [10.24432/C59K5F](https://doi.org/10.24432/C59K5F) |
| Citation | Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F |
| Download URL | `https://archive.ics.uci.edu/static/public/360/air+quality.zip` |
| Downloaded | 2026-10-23 22:37 UTC |
| Archive SHA-256 | `d4a64013fb385288a8a48d9d193ca7079b2e1bbddf6f8d458feb8c08ab2b8a2a` (1,543,989 bytes) |
| `AirQualityUCI.csv` SHA-256 | `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996` (785,065 bytes) |
| `AirQualityUCI.xlsx` SHA-256 | `c92d102bb35cd4c0b42c0b6a1336c4065ccdca7069f80a6890ee1b7e3124bd5d` (not used) |
| Licence | The UCI page lists CC BY 4.0, but its description says the data may be used exclusively for research. Raw data are therefore **not** committed. |

Reproduce with `python -m air_quality.data`. The command downloads the archive into `data/raw/` (git-ignored), checks both hashes, and prints the validation report. Any hash mismatch stops it with an error.

## 2. Reference file and raw format

**The CSV is the reference file.** The archive also contains an XLSX with the same 9,357 rows, the same timestamps, the same `-200` positions and an identical `CO(GT)` column.

The XLSX stores more decimal places. Sensor responses there are multiples of 0.25, while the CSV rounds them to integers (|difference| ≤ 0.5). `T`, `RH` and `C6H6(GT)` differ by ≤ 0.05. More stored digits do not imply higher measurement accuracy, and the difference is negligible against the sensors' range (roughly 220–2,800). We use the CSV because the project specification names it and it needs no extra dependency.

CSV format facts:

- ASCII, no BOM, CRLF line endings.
- `;` separates fields and `,` is the decimal mark.
- Every line ends with `;;`, producing two empty trailing columns.
- After 9,357 data rows the file contains 114 lines made only of `;`.
- `Date` is formatted `DD/MM/YYYY` and `Time` is formatted `HH.MM.SS`.

## 3. Differences from published metadata

| Item | Published | Actual file |
|---|---|---|
| Records | 9,358 | **9,357** |
| Period | March 2004 – February 2005 ("one year") | **2004-03-10 18:00 → 2005-04-04 14:00** (≈ 13 months) |
| Environmental column names on the UCI page | `Temperature`, `Relative Humidity` | `T`, `RH` (as in the project spec) |
| File structure | not described | 2 empty trailing columns, 114 trailing empty lines |

## 4. Schema and roles

Missing counts are after converting `-200` to NaN, out of 9,357 hours.

| Column | Role | Unit | Missing |
|---|---|---|---|
| `CO(GT)` | **target** | mg/m³ (resolution 0.1; range 0.1–11.9) | 1,683 (17.99%) |
| `PT08.S1(CO)` … `PT08.S5(O3)` | input | sensor response (no unit given) | 366 each |
| `T` | input | °C | 366 |
| `RH` | input | % | 366 |
| `AH` | input | not stated by UCI | 366 |
| `NMHC(GT)` | **forbidden** | µg/m³ | 8,443 (90.2%) |
| `C6H6(GT)` | **forbidden** | µg/m³ | 366 |
| `NOx(GT)` | **forbidden** | ppb | 1,639 |
| `NO2(GT)` | **forbidden** | µg/m³ | 1,642 |

Units are taken from the UCI variable table. The only missing-value marker found is `-200`. There are no other sentinel-like values, no native blanks inside data rows, and no negative values except 13 plausible winter temperatures (minimum −1.9 °C).

## 5. Loading rules (`load_raw`)

1. Read the file with `;` as separator and `,` as decimal mark. Drop the two empty columns and the fully blank rows.
2. Parse `Date` + `Time` with the strict format `%d/%m/%Y %H.%M.%S`. Loading fails on any duplicate timestamp or any timestamp not on the hour.
3. Convert `-200` to NaN in every numeric column.
4. Index the frame by `timestamp`, sorted, on the **complete hourly grid**. For this file the grid already has every hour, with no gaps or duplicates, so no rows are inserted.
5. **Keep every hour.** Rows without a target or without inputs stay in the frame, so that past-only features can later be computed on the true time grid. Filtering happens only through `eligible_mask`, after the split.

Timestamps are kept as naive local times exactly as recorded. On the three DST transition dates (2004-03-28, 2004-10-31, 2005-03-27) each day still has 24 hourly rows. **This does not establish whether or how DST was handled.** The source does not document a time-zone convention, and does not say whether a label marks the start or end of the averaging hour. This affects how hour-of-day features are interpreted, not the alignment between sensors and target, which share each timestamp.

## 6. Missingness structure

- **Target gaps are long and clustered.** There are 187 missing runs. The longest are 173 h (2004-10-13 → 10-20), 146 h (April 2004) and 142 h (October 2004). The monthly share of hours with a valid target ranges from 52.8% (October 2004) and 64.1% (August 2004) up to 97–99% (February–April 2005).
- **Input outages are all-or-nothing.** In 366 hours all five sensors and `T`/`RH`/`AH` are missing together, and no hour has only some inputs missing. The longest outage lasts 76 h (February 2005).
- Of those 366 input-missing hours, **330 still have a valid reference target** and 36 have no target either.

## 7. Eligibility rule (`eligible_mask`)

A row is eligible for training and primary evaluation if it has a valid `CO(GT)` **and** at least one current low-cost input. An hour with no current input cannot produce an estimate at prediction time. Excluded hours are not deleted: they stay on the grid and can still serve as past context for features.

Two separate quantities are reported, and they should not be confused:

- **Inputs available** (`inputs_available`): hours where the device produced at least one current reading, so an estimate is possible at prediction time. The target plays no role.
- **Eligible** (`eligible_mask`): inputs available **and** a valid reference target, so the hour can be used for training and scored in evaluation.

| Set | Grid hours | Inputs available | Valid target | No inputs (of which valid target) | **Eligible** | Eligible share of hours |
|---|---:|---:|---:|---:|---:|---:|
| Full period | 9,357 | 8,991 (0.961) | 7,674 | 366 (330) | **7,344** | 0.785 |
| Development | 7,494 | 7,214 (0.963) | 5,902 | 280 (245) | **5,657** | 0.755 |
| Final test | 1,863 | 1,777 (0.954) | 1,772 | 86 (85) | **1,687** | 0.906 |

The 330 hours with a valid target but no inputs (245 in development, 85 in the final test) are counted separately. They are the only rows lost to the input condition; all other ineligible hours lack the reference target.

## 8. Chronological split

- **`TEST_START = 2005-01-17 00:00:00`.** This is the midnight nearest to 80% of the timeline (2005-01-16 14:48). It is defined on time alone, before any target- or input-based filtering.
- **Development period:** 2004-03-10 18:00 → 2005-01-16 23:00, 7,494 hours. It includes **2005-01-01 → 01-16** (384 hours, 325 eligible).
- **Final test:** 2005-01-17 00:00 → 2005-04-04 14:00, 1,863 hours (19.9% of the timeline).
  - Because valid targets are more frequent late in the series, the test holds 23.0% of the eligible rows.
  - The test is used **once**, after features, models and preprocessing are frozen. It is never used for selection.
- **Seasonal caveat:** the test covers mid-January to early April 2005. The development period contains March–April 2004 and the first half of January 2005, but no February 2005. Any change in performance on the test cannot be attributed to sensor drift alone, because season and drift are confounded.

## 9. Time-aware cross-validation (`dev_folds`, `cv_indices`)

Folds use expanding windows. The fold boundaries are timestamps on the development period's complete hourly grid: the grid from `DATA_START` (2004-03-10 18:00) to `TEST_START` is cut into 6 equal blocks, and boundaries are rounded to midnight. Boundaries come from these constants, never from a filtered index, so no amount of row filtering can move them. Call `dev_folds()` with no arguments. Its overrides are keyword-only: `data_start` exists for unit tests on synthetic grids, and `gap_hours` for documented sensitivity checks. A positional call such as `dev_folds(X.index[0])` raises an error, because a feature-engineered frame that dropped its first 24 rows would otherwise shift every boundary. Eligibility is applied inside each fold afterwards. Every experiment uses these same folds, and therefore the same comparison samples.

| Fold | Train | Validation | Train hours (eligible) | Val hours (inputs available / eligible) | Val eligible share |
|---:|---|---|---:|---:|---:|
| 0 | 2004-03-10 18:00 → 2004-05-02 | 2004-05-02 → 2004-06-23 | 1,254 (983) | 1,248 (1,196 / 987) | 0.791 |
| 1 | 2004-03-10 18:00 → 2004-06-23 | 2004-06-23 → 2004-08-14 | 2,502 (1,970) | 1,248 (1,243 / 976) | 0.782 |
| 2 | 2004-03-10 18:00 → 2004-08-14 | 2004-08-14 → 2004-10-05 | 3,750 (2,946) | 1,248 (1,184 / 771) | 0.618 |
| 3 | 2004-03-10 18:00 → 2004-10-05 | 2004-10-05 → 2004-11-26 | 4,998 (3,717) | 1,248 (1,248 / 936) | 0.750 |
| 4 | 2004-03-10 18:00 → 2004-11-26 | 2004-11-26 → 2005-01-17 | 6,246 (4,653) | 1,248 (1,116 / 1,004) | 0.804 |

Intervals are half-open (`[start, end)`), and all boundaries are at 00:00. Fold 2 has the smallest eligible share because of the August–October 2004 target gaps, not because of sensor outages: inputs are available in 1,184 of its 1,248 hours.

**Using the folds with scikit-learn:** build `cv_indices` from the exact index of the `X`/`y` being passed, after every row filter, including any rows that a later step drops. Pass the result as `cv=` to `cross_validate`, which accepts it. `cross_val_predict` does **not** accept it: with expanding windows the first block is training-only, so the folds are not a partition. Produce out-of-fold predictions per fold instead.

### Gap decision: `GAP_HOURS = 0`

A gap between the training and validation windows prevents information from the validation period from entering training. In this design that cannot happen:

- **Features never look forward.** Planned features (Issue #3) use current and past low-cost inputs only (lags, past-only rolling windows, time of day). A training row's features therefore never contain validation-period values. A validation row may use training-period sensor readings, but those inputs are genuinely available at prediction time.
- **Features never use the target.** No feature is built from `CO(GT)`, so no label information crosses the boundary through features. A 24-hour lag does not call for a 24-hour gap: the lag reaches backwards into data that is legitimately available.
- **Learned preprocessing is fold-local.** Imputers and scalers are fitted inside each training fold (Issue #4).
- **It matches deployment and the final test.** In the calibration scenario a model fitted on all data up to a time *T* is applied from *T* onwards. The final test follows the development period without a gap, so a CV gap would evaluate a different situation from the final test.

The remaining effect is that validation hours just after the boundary sit close in time to the last training hours. This is the condition the deployed model would face, so it is not leakage. Each 52-day validation window is also long relative to any 24-hour feature window.

**Conditions for revisiting:** the decision must be reviewed if a feature ever uses future values, centred windows, or target-derived information. All of these are prohibited by the contract. `dev_folds(..., gap_hours=24)` is available for an optional robustness check, which would not change the primary protocol.

## 10. Leakage rules for downstream work

1. Model inputs come only from `INPUT_COLS` (`SENSOR_COLS + ENV_COLS`) plus timestamp-derived features. `FORBIDDEN_COLS` and `CO(GT)` are never inputs.
2. **Lag and rolling features are computed on the complete hourly grid returned by `load_raw()`, never on rows left after dropping missing values.** On the grid, `shift(k)` means exactly *k* hours earlier. After filtering it would silently mean "*k* rows earlier", bridging gaps that can last days. A lag that falls inside a missing period must be NaN, not the last value seen.
   - Windows look backwards only: `shift` before `rolling`, no centred windows, no interpolation or backfill from future values.
   - Computing features once on the full grid (development and test together) is safe because every window looks backwards. Development rows never see test-period values, and test rows only see their legitimate past.
3. Boundaries before filtering. `TEST_START` and the fold boundaries are fixed timestamps that do not depend on the data. Build past-only features on the full grid, split by time with `split_dev_test`, and only then filter rows with `eligible_mask`. Feature construction may come before or after the time split, because past-only windows give the same result either way. Row filtering must always come last.
4. Fit imputation, scaling and any learned transformation on training-fold rows only.
5. Analyses that relate inputs to the target for decision-making use the development period only.

## 11. Public interface (`air_quality.data`, `air_quality.split`)

| Name | Purpose |
|---|---|
| `download(raw_dir=None, force=False)` | Fetch and hash-verify the archive; return the CSV path |
| `load_raw(path=None)` | Load the CSV onto the complete hourly grid (`-200` → NaN, every hour kept) |
| `validate(df)` | Report on grid integrity, missingness and eligibility |
| `eligible_mask(df)`, `inputs_available(df)` | Row-eligibility rules |
| `TARGET`, `SENSOR_COLS`, `ENV_COLS`, `INPUT_COLS`, `FORBIDDEN_COLS` | Column roles |
| `TEST_START`, `split_dev_test(df)` | Final-test boundary and split |
| `DATA_START`, `N_SPLITS`, `GAP_HOURS`, `dev_folds()`, `Fold` | CV fold definition (call `dev_folds()` with defaults) |
| `cv_indices(index, folds)` | Positional train/validation indices for any (filtered) index; usable as `cv=` in `cross_validate` |
| `fold_summary(dev, folds)` | Per-fold dates, input availability and eligible counts |

Typical use:

```python
from air_quality.data import load_raw, eligible_mask, INPUT_COLS, TARGET
from air_quality.split import split_dev_test, dev_folds, cv_indices

df = load_raw()                                 # complete hourly grid
# features = build_features(df, ...)            # Issue #3: on the full grid, before filtering
dev, test = split_dev_test(df)                  # time-only boundary; test stays locked
dev_ok = dev[eligible_mask(dev)]                # filter after splitting
cv = cv_indices(dev_ok.index, dev_folds())      # identical folds for every experiment
```
