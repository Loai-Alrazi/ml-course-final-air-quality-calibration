# Time-Aware Air Quality Sensor Calibration: CO Estimation and Drift Evaluation

**معايرة مستشعرات جودة الهواء باستخدام تعلم الآلة**

> **Status:** Data foundation in place (Issue #1): verified dataset, data contract, loading, eligibility rules, chronological split and CV folds. EDA (#2), feature engineering (#3) and modeling (#4) have not started; there are no model results yet.

## Problem

Estimate the hourly reference carbon monoxide concentration `CO(GT)` (mg/m³) for the **current hour** from low-cost metal-oxide sensor readings and environmental measurements. This is current-hour sensor calibration, not next-hour or next-day forecasting.

**Research question**

> Can a multi-sensor calibration model, supported by justified feature engineering, estimate reference CO concentrations on future chronological data more accurately than a simple linear calibration of the CO-targeted sensor alone?

This is an academic project. Its models must not be presented as a certified replacement for regulatory monitoring instruments.

## Dataset

- **Source:** UCI Machine Learning Repository — [Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality)
- **DOI:** [10.24432/C59K5F](https://doi.org/10.24432/C59K5F)
- **Citation:** Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59K5F
- **Verified file:** 9,357 hourly records from 2004-03-10 18:00 to 2005-04-04 14:00 on a complete hourly grid, from one Italian roadside location; missing values tagged as `-200`. The published description (9,358 records, March 2004 – February 2005) differs slightly; see [docs/data_contract.md](docs/data_contract.md).
- **Licensing note:** the UCI page lists CC BY 4.0, while the dataset description states research-only use. To stay conservative, **raw data is not committed to this repository**; it is downloaded and hash-verified locally (see Setup).

## Work plan

| Issue | Owner | Area |
|---|---|---|
| [#1 Project Foundation & Data Contract](../../issues/1) | Loai-Alrazi (Project Lead) | Dataset verification, data contract, loading, chronological split |
| [#2 EDA & Data Quality](../../issues/2) | EmranYaseen-ALSabry | Focused EDA and data-quality audit |
| [#3 Feature Engineering & Leakage Tests](../../issues/3) | ZiyadAhmedAmmar | Causal features and leakage tests |
| [#4 Modeling, Validation & Evaluation](../../issues/4) | Mohammed-Ahmed-Alansi | Baselines, time-aware CV, ablation, evaluation |

Execution order: #1 first. After #1 is merged, #2 and #3 can proceed in parallel, and #4 can start the baselines. Feature-engineering experiments in #4 wait for #3. The final chronological test is run once, by the Project Lead, after model and feature decisions are frozen.

Team submission has been approved by the instructor.

## Repository layout

```
├── docs/
│   ├── PROJECT_SPEC.md          # binding project contract
│   ├── problem_definition.md    # Issue #1
│   ├── data_contract.md         # Issue #1
│   ├── data_audit.md            # Issue #2
│   └── feature_availability.md  # Issue #3
├── notebooks/01_eda.ipynb       # Issue #2
├── reports/
│   ├── figures/eda/             # Issue #2
│   ├── figures/modeling/        # Issue #4
│   └── model_results.md         # Issue #4
├── src/air_quality/
│   ├── data.py, split.py        # Issue #1
│   ├── features.py              # Issue #3
│   └── modeling.py, evaluation.py  # Issue #4
└── tests/                       # one test module per owner
```

Files listed above are created by their owning issue; only the package skeleton exists at bootstrap.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Download and verify the dataset (saved to the git-ignored `data/raw/`), then run the tests:

```bash
python -m air_quality.data
pytest -q
```

The data contract, chronological split, and CV folds are documented in [docs/data_contract.md](docs/data_contract.md). Experiment commands will be added as the corresponding issues are completed.

## Contributing

- Work on a branch named `issue-<N>-<short-name>` and open a pull request that references its issue (`Closes #N`).
- Change only the files owned by your issue; request changes to shared code through its owner.
- `pytest -q` must pass before review. `main` is protected: changes arrive only through pull requests, and each needs the Project Lead's approval (code owner) before merging.
- Never commit raw data, credentials, or tokens.

See [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) for the full contract.
