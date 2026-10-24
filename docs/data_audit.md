# Data Quality Audit & Exploratory Data Analysis Report

**Project:** Time-Aware Air Quality Sensor Calibration: CO Estimation and Drift Evaluation  
**Issue:** #2 — EDA & Data Quality  
**Author:** Emran Yaseen Al-Sabry (`EmranYaseen-ALSabry`)  
**Status:** Complete  
**Date:** 2026-10-24  
**Data Contract Version:** v1.1  

---

## 1. Executive Summary & Objective

This data audit establishes the empirical foundation for all downstream feature engineering (Issue #3) and model training (Issue #4). In strict accordance with `docs/data_contract.md` (v1.1) and `docs/problem_definition.md`:

1. **Zero Data Leakage:** Any exploratory analysis, correlation evaluation, or distribution profiling linking input features to the target `CO(GT)` is conducted strictly on the **Development Period** (`2004-03-10 18:00` to `2005-01-16 23:00`). The **Final Test Period** (`TEST_START = 2005-01-17 00:00`) remains strictly isolated.
2. **Defensible Reporting:** All reported findings are based on exact measured quantities verified against the official UCI Air Quality raw CSV.
3. **No Unwarranted Drift Claims:** We distinguish descriptive distribution shifts between seasons from proven sensor aging/drift, as seasonal weather variation and sensor drift are inextricably confounded.

---

## 2. Dataset Verification & Hourly Grid Integrity

The raw dataset was ingested and verified via `air_quality.data.load_raw()`:

* **Source:** UCI Machine Learning Repository — Air Quality Dataset (DOI: `10.24432/C59K5F`).
* **Raw CSV SHA-256:** `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`.
* **Total Timeline Span:** `2004-03-10 18:00:00` to `2005-04-04 14:00:00` (approximately 13 months).
* **Total Grid Hours:** Exactly **9,357** consecutive hourly slots.
* **Grid Completeness:** 0 missing hourly timestamps, 0 duplicate timestamps, and 0 misalignment errors.
* **Missing Value Sentinel:** `-200` is the sole missing value marker across all numeric columns, converted cleanly to `np.nan`.

---

## 3. Missingness & Outage Architecture

The missingness profile across the dataset was audited thoroughly:

### 3.1 Column-Wise Missingness Summary

| Column | Role in Project | Missing Hours | Missing % | Notes |
|:---|:---|:---:|:---:|:---|
| **`CO(GT)`** | **Target** | **1,683** | **17.99%** | Reference analyzer ground truth |
| `PT08.S1(CO)` | Low-Cost Input | 366 | 3.91% | Tin-oxide CO sensor |
| `PT08.S2(NMHC)` | Low-Cost Input | 366 | 3.91% | Titania NMHC sensor |
| `PT08.S3(NOx)` | Low-Cost Input | 366 | 3.91% | Tungsten-oxide NOx sensor |
| `PT08.S4(NO2)` | Low-Cost Input | 366 | 3.91% | Tungsten-oxide NO2 sensor |
| `PT08.S5(O3)` | Low-Cost Input | 366 | 3.91% | Indium-oxide O3 sensor |
| `T` | Environmental Input | 366 | 3.91% | Ambient temperature (°C) |
| `RH` | Environmental Input | 366 | 3.91% | Relative humidity (%) |
| `AH` | Environmental Input | 366 | 3.91% | Absolute humidity |
| `NMHC(GT)` | **Forbidden** | 8,443 | 90.23% | Reference analyzer; excluded |
| `C6H6(GT)` | **Forbidden** | 366 | 3.91% | Reference analyzer; excluded |
| `NOx(GT)` | **Forbidden** | 1,639 | 17.52% | Reference analyzer; excluded |
| `NO2(GT)` | **Forbidden** | 1,642 | 17.55% | Reference analyzer; excluded |

### 3.2 Key Missingness Properties

1. **All-or-Nothing Input Sensor Outages:**
   * Across the 9,357 hours, all 5 low-cost sensors and all 3 environmental variables (`T`, `RH`, `AH`) are missing **simultaneously in exactly 366 hours**.
   * There are **0 hours with partial input sensor availability**. If any low-cost sensor is active, all 8 input columns are present.
   * Input outages occur across **16 discrete outage events**. The longest continuous input outage lasts **76 consecutive hours** (observed in February 2005 during the test period), followed by runs of 75h, 52h, 45h, and 38h.
2. **Clustered Reference Target Gaps:**
   * The reference analyzer `CO(GT)` suffers **1,683 missing hours** distributed across **187 separate outage events**.
   * The longest continuous target outages are:
     * **173 hours:** 2004-10-13 to 2004-10-20.
     * **146 hours:** April 2004.
     * **142 hours:** October 2004.
     * **122 hours:** August 2004.
     * **96 hours:** December 2004.
3. **Cross-Availability Breakdown:**
   * Hours where inputs are absent but reference target is present: **330 hours** (245 in dev, 85 in test).
   * Hours where both inputs and target are absent: **36 hours** (35 in dev, 1 in test).

Visualized in: `reports/figures/eda/01_missingness_timeline.png`.

---

## 4. Chronological Split & Eligibility Partitioning

Eligibility for training and primary evaluation is governed strictly by:
$$\text{eligible} = (\text{CO(GT) is not NaN}) \land (\text{inputs\_available})$$

| Partition | Grid Hours | Inputs Available | Valid Target | Both Present (**Eligible**) | Eligible Share |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Full Dataset** | 9,357 | 8,991 (96.09%) | 7,674 (82.01%) | **7,344** | **78.49%** |
| **Development Period** (`< 2005-01-17`) | 7,494 | 7,214 (96.26%) | 5,902 (78.76%) | **5,657** | **75.49%** |
| **Final Test Period** (`≥ 2005-01-17`) | 1,863 | 1,777 (95.38%) | 1,772 (95.12%) | **1,687** | **90.55%** |

> **Critical Note:** The final test period has a significantly higher proportion of valid target observations (90.55% vs. 75.49%) because the reference analyzer was considerably more reliable in early 2005 than in autumn 2004.

---

## 5. Target Variable Analysis (`CO(GT)`) in Development Period

*All statistics below are computed strictly on the 5,657 eligible hours of the development period.*

### 5.1 Distribution Summary Statistics

| Metric | Measured Value | Interpretation |
|:---|:---:|:---|
| **Count** | 5,657 | Evaluated sample size |
| **Mean** | 2.19 mg/m³ | Average reference CO concentration |
| **Median** | 1.90 mg/m³ | Central tendency (less sensitive to peaks) |
| **Std Dev** | 1.46 mg/m³ | Substantial hourly dispersion |
| **Min** | 0.10 mg/m³ | Sensor detection limit / clean air |
| **25th Percentile** | 1.10 mg/m³ | Lower quartile |
| **75th Percentile** | 2.90 mg/m³ | Upper quartile |
| **95th Percentile** | 5.00 mg/m³ | Heavy pollution threshold |
| **Max** | 11.90 mg/m³ | Peak pollution event |
| **Skewness** | **+1.36** | Moderately positive (right-skewed tail) |

The distribution follows an approximate log-normal shape typical of urban air pollutants, where background concentrations dominate but traffic and atmospheric stagnation induce severe episodic spikes.

### 5.2 Monthly Evolution Across the Development Period

| Month | Eligible Hours | Mean CO(GT) (mg/m³) | Median CO(GT) (mg/m³) | Std Dev (mg/m³) |
|:---|:---:|:---:|:---:|:---:|
| **2004-03** | 492 | 2.29 | 2.10 | 1.35 |
| **2004-04** | 468 | 2.30 | 2.00 | 1.44 |
| **2004-05** | 592 | 1.94 | 1.80 | 1.20 |
| **2004-06** | 607 | 1.93 | 1.80 | 1.15 |
| **2004-07** | 558 | 1.78 | 1.60 | 1.10 |
| **2004-08** | 428 | **1.25** | **1.20** | **0.58** |
| **2004-09** | 555 | 2.16 | 1.80 | 1.43 |
| **2004-10** | 392 | **2.72** | **2.50** | 1.64 |
| **2004-11** | 679 | **2.69** | 2.30 | **1.93** |
| **2004-12** | 561 | 2.58 | 2.30 | 1.65 |
| **2005-01** (dev slice) | 325 | 2.45 | 2.20 | 1.25 |

**Findings:**
* **Summer minimum in August (mean 1.25 mg/m³):** Corresponds to the Italian summer vacation period (*Ferragosto*), characterized by sharply reduced vehicular traffic in Rome, combined with strong atmospheric convection and high planetary boundary layer heights.
* **Winter maximum in October–November (mean 2.72 mg/m³, max std 1.93):** Driven by residential heating, increased commuter traffic, and frequent thermal inversions trapping pollutants near the roadside.

Visualized in: `reports/figures/eda/02_target_distribution_and_monthly.png`.

---

## 6. Sensor-to-Target Relationship & Temporal Stability

### 6.1 Calibration Baseline: `PT08.S1(CO)` vs `CO(GT)`

* **Pearson Correlation coefficient ($r$):** **0.8931** ($R^2 \approx 0.798$)
* **Spearman Rank Correlation ($\rho$):** **0.8999**
* **OLS Linear Calibration Fit:**
  $$\widehat{\text{CO(GT)}} = 0.0053 \times \text{PT08.S1(CO)} - 3.70$$

**Modulation by Ambient Temperature:**
The scatter plot reveals that for identical `PT08.S1(CO)` sensor response levels, higher ambient temperatures ($T > 25^\circ\text{C}$, summer) systematically shift reference `CO(GT)` downward relative to colder conditions ($T < 10^\circ\text{C}$, winter). Metal-oxide sensor conductivity depends fundamentally on surface oxygen ion sorption, which is strongly temperature- and moisture-dependent.

Visualized in: `reports/figures/eda/03_target_sensor_relationship_dev.png`.

### 6.2 Temporal Stability (Rolling Correlation)

To test the stability of the relationship over time, 14-day (336h) and 30-day (720h) rolling Pearson correlations were computed over the development period:

| Rolling Window | Mean | Median | Std Dev | Min | Max |
|:---|:---:|:---:|:---:|:---:|:---:|
| **14-day Window** | **0.915** | 0.925 | 0.035 | 0.398 | 1.000 |
| **30-day Window** | **0.907** | 0.910 | 0.034 | 0.398 | 1.000 |

**Stability Observation:**
* The sensor-target coupling is remarkably strong ($r > 0.88$) across Spring, Summer, and early Autumn.
* A pronounced dip occurs in late Autumn (November 2004), where correlation briefly declines toward ~0.70 (and dips locally to ~0.40 on narrow windows). This drop coincides with the sharp onset of cold weather, heavy precipitation, and rapid atmospheric transitions.

Visualized in: `reports/figures/eda/04_rolling_correlation_stability.png`.

---

## 7. Descriptive Sensor Shifts: Early vs. Late Period

We compared the distributions of all 5 low-cost sensors between:
* **Early Development Period:** `2004-03-11` to `2004-05-11` (Spring 2004, 1,464 grid hours).
* **Late Development Period:** `2004-11-15` to `2005-01-16` (Winter 2004–2005, 1,488 grid hours).

### 7.1 Measured Sensor Mean Shifts

| Sensor Column | Target Gas | Early Mean | Late Mean | Absolute Δ | Percentage Shift |
|:---|:---|:---:|:---:|:---:|:---:|
| `PT08.S1(CO)` | CO-targeted | 1,168.2 | 1,137.3 | -30.9 | **-2.64%** |
| `PT08.S2(NMHC)` | NMHC-targeted | 943.1 | 962.3 | +19.2 | **+2.04%** |
| `PT08.S3(NOx)` | NOx-targeted | 963.7 | 801.9 | -161.7 | **-16.78%** |
| `PT08.S4(NO2)` | NO2-targeted | 1,606.4 | 1,266.2 | -340.2 | **-21.18%** |
| `PT08.S5(O3)` | O3-targeted | 994.9 | 1,176.2 | +181.4 | **+18.23%** |

### 7.2 Crucial Scientific Distinction: Drift vs. Seasonality

> [!WARNING]
> **Sensor drift cannot be claimed as proven.**  
> Although `PT08.S3(NOx)` and `PT08.S4(NO2)` exhibit large downward shifts (-16.8% and -21.2%) while `PT08.S5(O3)` shifts upward (+18.2%), these temporal differences **cannot be attributed solely to sensor aging or drift**.
>
> 1. **Extreme Seasonal Confounding:** Ambient temperature dropped from a mean of $16.4^\circ\text{C}$ in the early period to $8.7^\circ\text{C}$ in the late period, and absolute humidity dropped from $0.84$ to $0.68$. Metal oxide resistance is known to vary nonlinearly with atmospheric temperature and moisture.
> 2. **Atmospheric Chemistry:** Photochemical ozone ($O_3$) production and nitrogen oxide ($NO_x / NO_2$) equilibrium shift naturally between summer and winter.
>
> Downstream models must evaluate whether nonlinear environmental compensation resolves these shifts before invoking irreversible physical sensor drift.

Visualized in: `reports/figures/eda/05_sensor_distribution_early_vs_late.png`.

---

## 8. Environmental Patterns & Diurnal Cycles

* **Seasonal Progression:** Temperature ranges from a maximum of $44.6^\circ\text{C}$ in summer to a minimum of $-1.9^\circ\text{C}$ in winter.
* **Diurnal Traffic Bimodal Peaks:** Diurnal profiles of `CO(GT)` and low-cost sensor readings exhibit two stark roadside traffic rush peaks:
  * **Morning Rush Hour:** 08:00 – 09:00 local time.
  * **Evening Rush Hour:** 19:00 – 20:00 local time.
  * **Night Minimum:** 03:00 – 05:00 local time.

Visualized in: `reports/figures/eda/06_environmental_seasonal_patterns.png`.

---

## 9. Actionable Recommendations for Downstream Issues

### 9.1 For Feature Engineering (Issue #3 — ZiyadAhmedAmmar)
1. **Handling the 366-Hour Sensor Outages:**
   Because low-cost input outages are all-or-nothing and last up to 76 hours, any autoregressive lag feature (e.g. $t-1, t-2$) or rolling window calculation across outage boundaries will propagate NaNs. Consider using causal zero-leakage forward fill or masking indicators inspired by SmartImpute methodology for lags $\le 3$ hours, but leave multi-day gaps uncomputed.
2. **Sensor-Environment Interactions:**
   Given the strong modulation of `PT08.S1(CO)` by temperature and humidity, create explicit interaction terms (e.g. $\text{PT08.S1} \times T$, $\text{PT08.S1} \times \text{AH}$).
3. **Diurnal and Day-of-Week Features:**
   Cyclical encoding ($\sin / \cos$) of the hour of the day (24h) and day of the week (7d) is strongly justified by the morning and evening traffic peaks and weekend traffic reductions.

### 9.2 For Modeling & Validation (Issue #4 — Mohammed-Ahmed-Alansi)
1. **Target Skewness:**
   With skewness $= +1.36$, linear models may be disproportionately influenced by high-pollution tail events. The primary evaluation metric is **MAE**, which provides robustness against extreme outliers compared to RMSE.
2. **Single vs Multi-Sensor Calibration:**
   While `PT08.S1(CO)` alone achieves $r \approx 0.893$, the presence of other sensors (`PT08.S2` through `PT08.S5`) and environmental inputs will likely help isolate cross-gas sensitivity and temperature interference, which is the core thesis of Experiment C vs B.
3. **Validation Windows:**
   Follow the 3-fold expanding cross-validation protocol pinned in contract v1.1 (`dev_folds()`). Fold 0 (August–October 2004) will test summer-to-autumn transition, while Fold 2 (November 2004–January 2005) tests winter calibration stability.

---

## 10. Audit Artifacts & Figure Registry

All generated figures conform to publication quality (300 DPI, styled palettes, explicit units) and are stored in `reports/figures/eda/`:

1. [`reports/figures/eda/01_missingness_timeline.png`](file:///c:/Users/OMRRAN/Desktop/Cource%20AI/%234-%20ML/Competitions/ml-course-final-air-quality-calibration/reports/figures/eda/01_missingness_timeline.png) — Binary missingness matrix across all 13 columns, annotating 173h target and 76h sensor outages and `TEST_START`.
2. [`reports/figures/eda/02_target_distribution_and_monthly.png`](file:///c:/Users/OMRRAN/Desktop/Cource%20AI/%234-%20ML/Competitions/ml-course-final-air-quality-calibration/reports/figures/eda/02_target_distribution_and_monthly.png) — Distribution density/KDE of `CO(GT)` and monthly boxplots in development period.
3. [`reports/figures/eda/03_target_sensor_relationship_dev.png`](file:///c:/Users/OMRRAN/Desktop/Cource%20AI/%234-%20ML/Competitions/ml-course-final-air-quality-calibration/reports/figures/eda/03_target_sensor_relationship_dev.png) — Scatter plot with linear fit and temperature-color gradient on development partition.
4. [`reports/figures/eda/04_rolling_correlation_stability.png`](file:///c:/Users/OMRRAN/Desktop/Cource%20AI/%234-%20ML/Competitions/ml-course-final-air-quality-calibration/reports/figures/eda/04_rolling_correlation_stability.png) — 14-day and 30-day rolling Pearson correlation between `PT08.S1(CO)` and `CO(GT)`.
5. [`reports/figures/eda/05_sensor_distribution_early_vs_late.png`](file:///c:/Users/OMRRAN/Desktop/Cource%20AI/%234-%20ML/Competitions/ml-course-final-air-quality-calibration/reports/figures/eda/05_sensor_distribution_early_vs_late.png) — Comparative distribution curves for all 5 low-cost sensors (Spring 2004 vs Winter 2004–2005).
6. [`reports/figures/eda/06_environmental_seasonal_patterns.png`](file:///c:/Users/OMRRAN/Desktop/Cource%20AI/%234-%20ML/Competitions/ml-course-final-air-quality-calibration/reports/figures/eda/06_environmental_seasonal_patterns.png) — Seasonal evolution of temperature/humidity and diurnal rush-hour profiles.
