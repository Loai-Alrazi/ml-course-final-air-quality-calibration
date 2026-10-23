# PROJECT_SPEC.md

**Project:** Time-Aware Air Quality Sensor Calibration: CO Estimation and Drift Evaluation  
**Arabic title:** معايرة مستشعرات جودة الهواء باستخدام تعلم الآلة  
**Version:** 1.0  
**Date:** 2026-10-24  
**Deadline:** Saturday, October 24, 2026  
**Project Lead:** Loai-Alrazi

---

## 1. Project Objective

Build a scientifically defensible, reproducible, end-to-end machine learning project that estimates hourly reference carbon monoxide (CO) concentration from low-cost air-quality sensor readings.

The objective is **current-hour sensor calibration**, not next-hour or next-day forecasting.

The central research question is:

> Can a multi-sensor calibration model, supported by justified feature engineering, estimate reference CO concentrations on future chronological data more accurately than a simple linear calibration of the CO-targeted sensor alone?

The project must answer this question through controlled experiments rather than assumed improvements.

## 2. Dataset

**Official source:** UCI Machine Learning Repository — Air Quality Dataset

https://archive.ics.uci.edu/dataset/360/air+quality

**DOI:** 10.24432/C59K5F

According to the published dataset description, it contains approximately 9,358 hourly observations collected between March 2004 and February 2005 at one Italian roadside monitoring location.

The dataset includes readings from five metal-oxide sensors, environmental measurements, and reference gas concentrations.

**Important:** Published dataset characteristics must be verified against the actual downloaded file before implementation decisions are finalized.

The original research report is supporting context, not a substitute for checking the official source and raw file.

## 3. Problem Definition

**Task:** Supervised regression.

**Target:** `CO(GT)` — hourly reference CO concentration in mg/m³.

**Prediction time:** After the current hour's low-cost sensor readings are available.

**Potential beneficiaries:** Air-quality monitoring researchers and teams evaluating low-cost sensing systems.

**Deployment limitation:** This academic model must not be presented as a certified replacement for regulatory monitoring instruments.

## 4. Allowed and Disallowed Inputs

Candidate permitted inputs:

- `PT08.S1(CO)`
- `PT08.S2(NMHC)`
- `PT08.S3(NOx)`
- `PT08.S4(NO2)`
- `PT08.S5(O3)`
- `T`
- `RH`
- `AH`
- Time features derived from `Date` and `Time`

Do not use the following reference measurements as predictors:

- `NMHC(GT)`
- `C6H6(GT)`
- `NOx(GT)`
- `NO2(GT)`

Do not use current or previous `CO(GT)` values as input features in the core low-cost-only scenario.

All input features must be available at the declared prediction time.

## 5. Data Validation Requirements

Before modeling:

1. Download the official dataset.
2. Inspect the actual CSV structure and encoding.
3. Verify separators, decimal formatting, and empty columns.
4. Parse and validate timestamps.
5. Sort records chronologically.
6. Inspect duplicates, gaps, missing values, and invalid values.
7. Convert the documented `-200` missing-value marker appropriately.
8. Verify the number of valid target observations.
9. Document any differences between the actual file and published metadata.
10. Record the dataset source and reproducibility information.

Do not invent missing-value handling or assume the published schema matches the downloaded file perfectly.

## 6. Evaluation Protocol

- Use a chronological holdout for the final test period, approximately the last 20% of the timeline, subject to data audit.
- Define the temporal boundary before target-dependent filtering.
- Reserve the final test for one evaluation after model and feature selection.
- Use time-aware cross-validation on the development period.
- Use identical validation folds for comparable experiments.
- Fit imputers, scalers, and other learned transformations on training folds only.
- Never use future observations to construct historical features.
- Avoid random train/test splitting for the final scientific claim.
- Evaluate whether a temporal gap is justified based on feature windows and data continuity.

## 7. Modeling Scope

Required baselines:

1. DummyRegressor.
2. Single-sensor linear or Ridge calibration using `PT08.S1(CO)`.

Candidate improvements:

3. Multi-sensor Ridge Regression.
4. Ridge with justified engineered features.
5. One nonlinear model, preferably Random Forest or HistGradientBoosting.

Do not train many models or run large hyperparameter searches.

**Primary metric:** MAE (mg/m³).

**Secondary metrics:** RMSE and R².

Do not report expected performance numbers before running experiments.

## 8. Feature Engineering

Potential experiments:

- Cyclical hour representation.
- Day-of-week indicators.
- Environmental interactions such as temperature × relative humidity.
- Sensor lag features using past observations.
- Past-only rolling statistics.
- Missingness indicators where justified.

Each engineered feature must have:

- A clear motivation.
- Identified source columns.
- A defined availability time.
- A leakage assessment.
- A controlled comparison against the previous feature set.

No feature is assumed useful before evaluation.

## 9. Required Experiments

At minimum, compare:

**A.** Dummy baseline.

**B.** Single-sensor Ridge.

**C.** Multi-sensor Ridge.

**D.** Multi-sensor Ridge with selected engineered features.

**E.** Best justified feature set with one nonlinear model.

Compare experiments using the same time-aware validation protocol.

The final model must be selected based on development-period evidence, not final-test performance.

## 10. Error Analysis

Analyze:

- Prediction errors over time.
- Performance on high-CO observations.
- Errors under different temperature and humidity conditions.
- Effects of missing sensor readings.
- Differences between validation and final-test performance.
- Possible temporal degradation.

Do not claim that sensor drift has been proven to cause performance deterioration merely because late-period errors increase.

## 11. Team and Ownership

The project has four proposed contributors, subject to instructor approval for team submission.

**Issue 01 — Loai-Alrazi (Project Lead)**  
Problem framing, dataset verification, data contract, ingestion, chronological split, and integration oversight.

Owned implementation areas:
- `src/air_quality/data.py`
- `src/air_quality/split.py`
- `docs/problem_definition.md`
- `docs/data_contract.md`
- Related tests

**Issue 02 — EmranYaseen-ALSabry**  
Exploratory data analysis, missingness investigation, descriptive temporal analysis, and visualizations.

Owned areas:
- `notebooks/01_eda.ipynb`
- `docs/data_audit.md`
- `reports/figures/`

**Issue 03 — ZiyadAhmedAmmar**  
Feature engineering, feature availability documentation, and tests preventing temporal leakage.

Owned areas:
- `src/air_quality/features.py`
- `docs/feature_availability.md`
- `tests/test_features.py`

**Issue 04 — Mohammed-Ahmed-Alansi**  
Baselines, modeling pipelines, time-aware cross-validation, ablation comparisons, and evaluation reporting.

Owned areas:
- `src/air_quality/modeling.py`
- `src/air_quality/evaluation.py`
- `reports/model_results.md`
- Related modeling tests

The project lead owns final integration, README approval, and the authorization to unlock final-test evaluation.

Issue numbers express creation order and priority, not a requirement that all work must be performed strictly sequentially.

Issue 01 establishes the shared data and evaluation contracts. Issues 02 and 03 may proceed in parallel afterward. Issue 04 can prepare baselines after Issue 01 but must wait for the approved feature interface before final ablation experiments.

## 12. Minimum Deliverables

The final repository must contain:

- Clear README.
- Official dataset source and citation.
- Reproducible environment configuration.
- Data loading and validation code.
- Focused EDA.
- Feature engineering with leakage safeguards.
- Preprocessing and modeling pipelines.
- Baseline and model comparison results.
- Time-aware cross-validation.
- Final chronological test results.
- Error analysis.
- A concise limitations section.
- Requirements and execution instructions.

Do not add a web application, API, Docker, cloud infrastructure, or deep learning unless explicitly approved after the scientific core is complete.

## 13. Repository Rules

- Create the GitHub repository under the verified project leader account.
- Preferred repository name: `air-quality-co-calibration`.
- Use a private repository initially unless the project lead explicitly approves public visibility.
- Do not commit raw dataset files unless their redistribution rights are clearly established.
- Ignore virtual environments, credentials, secrets, local caches, and raw data.
- Prefer small, reviewable pull requests.
- Do not force-push or overwrite unrelated existing files.
- Do not merge changes without review.
- Keep all four contributors' responsibilities visible in GitHub Issues.
- Assign issues only when GitHub permissions permit; otherwise record the responsible GitHub handle in the issue description.
- Obtain approval before inviting collaborators or changing repository visibility.

## 14. Definition of Done

The project is complete when:

1. The actual dataset has been inspected and validated.
2. The prediction task and allowed features are documented.
3. The final test has remained isolated from model selection.
4. A practical baseline has been implemented.
5. At least one controlled feature-engineering comparison has been completed.
6. Models have been evaluated using a suitable temporal protocol.
7. Final results and limitations are documented honestly.
8. The repository can be reproduced from its instructions.
9. All four contributors understand the scientific decisions.
10. The project lead approves the final submission.

## 15. Scope and Decision Policy

Scientific correctness takes priority over complexity.

When time is limited:

- Prefer fewer, completed experiments.
- Prefer reliable baselines.
- Prefer clear evidence over additional features.
- Reduce optional scope before weakening evaluation validity.
- Do not invent results or conceal unsuccessful experiments.

**No delegated coding agent may autonomously expand the project scope beyond this specification.**

This specification is the primary project contract. The deep-research report is supporting evidence. The official dataset and its actual downloaded contents are the final authority on data structure.