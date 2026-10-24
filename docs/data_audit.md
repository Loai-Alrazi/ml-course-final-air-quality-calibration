# Data Quality Audit & Exploratory Data Analysis Report

**Project:** Time-Aware Air Quality Sensor Calibration: CO Estimation and Drift Evaluation  
**Issue:** #2 — EDA & Data Quality  
**Author:** Emran Yaseen Al-Sabry (`EmranYaseen-ALSabry`)  
**Status:** Complete (Revised per Project Lead Review)  
**Date:** 2026-10-24  
**Data Contract Version:** v1.1 (`docs/data_contract.md`)  

---

## 1. Executive Summary & Objective

This data audit establishes the empirical foundation for downstream feature engineering (Issue #3) and model training (Issue #4). In strict accordance with `docs/data_contract.md` (v1.1) and `docs/problem_definition.md`:

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
   * There are **0 hours with partial input sensor availability**.
   * Input outages occur across **16 discrete outage events**. The longest continuous input outage lasts **76 consecutive hours** (observed in February 2005 during the test period), followed by runs of 75h, 52h, 45h, and 38h.
2. **Clustered Reference Target Gaps:**
   * The reference analyzer `CO(GT)` suffers **1,683 missing hours** distributed across **187 separate outage episodes**.
   * The longest continuous target outages are:
     * **173 hours:** 2004-10-13 11:00 → 2004-10-20 15:00 (October 2004).
     * **146 hours:** 2004-04-17 03:00 → 2004-04-23 04:00 (April 2004).
     * **142 hours:** 2004-10-01 18:00 → 2004-10-07 15:00 (October 2004).
     * **122 hours:** 2004-07-26 03:00 → 2004-07-31 04:00 (late July 2004).
     * **96 hours:** 2004-12-28 01:00 → 2005-01-01 00:00 (late December 2004).
3. **Monthly Target Availability Variation:**
   * The proportion of valid `CO(GT)` hours ranges widely: October 2004 (52.8%) and August 2004 (64.1%) have high missingness, while February–April 2005 reaches 97–99% completeness.
4. **Cross-Availability Breakdown:**
   * Hours where inputs are absent but reference target is present: **330 hours** (245 in dev, 85 in test).
   * Hours where both inputs and target are absent: **36 hours** (35 in dev, 1 in test).

Visualized in: `../reports/figures/eda/01_missingness_timeline.png`.

---

## 4. Chronological Split & Eligibility Partitioning

Eligibility for training and primary evaluation is governed strictly by:
$$\text{eligible} = (\text{CO(GT) is not NaN}) \land (\text{inputs\_available})$$

See `docs/data_contract.md` §7 for the formal protocol definition.

| Partition | Grid Hours | Inputs Available | Valid Target | Both Present (**Eligible**) | Eligible Share |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Full Dataset** | 9,357 | 8,991 (96.09%) | 7,674 (82.01%) | **7,344** | **78.49%** |
| **Development Period** (`< 2005-01-17`) | 7,494 | 7,214 (96.26%) | 5,902 (78.76%) | **5,657** | **75.49%** |
| **Final Test Period** (`≥ 2005-01-17`) | 1,863 | 1,777 (95.38%) | 1,772 (95.12%) | **1,687** | **90.55%** |

---

## 5. Target Variable Analysis (`CO(GT)`) in Development Period

*All statistics below are computed strictly on the 5,657 eligible hours of the development period.*

### 5.1 Distribution Summary Statistics

| Metric | Measured Value | Interpretation |
|:---|:---:|:---|
| **Count** | 5,657 | Evaluated sample size |
| **Mean** | 2.19 mg/m³ | Average reference CO concentration |
| **Median** | 1.90 mg/m³ | Central tendency |
| **Std Dev** | 1.46 mg/m³ | Hourly dispersion |
| **Min** | 0.10 mg/m³ | Lower boundary |
| **25th Percentile** | 1.10 mg/m³ | Lower quartile |
| **75th Percentile** | 2.90 mg/m³ | Upper quartile |
| **95th Percentile** | **4.90 mg/m³** | Upper 5% threshold |
| **Max** | 11.90 mg/m³ | Maximum recorded concentration |
| **Skewness** | **+1.36** | Moderately positive (right-skewed tail) |

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

**Descriptive Observations:**
* August 2004 exhibits the lowest monthly mean (**1.25 mg/m³**, std 0.58).
* October and November 2004 exhibit the highest monthly means (**2.72 mg/m³** and **2.69 mg/m³**, std up to 1.93).
* *Note on potential causes:* Hypotheses regarding holiday activity reductions or meteorological dispersion conditions remain unverified, as external traffic and planetary boundary layer data are not present in this dataset.

Visualized in: `../reports/figures/eda/02_target_distribution_and_monthly.png`.

---

## 6. Sensor-to-Target Relationship & Temporal Stability

### 6.1 Calibration Baseline: `PT08.S1(CO)` vs `CO(GT)`

* **Sample Size ($N$):** 5,657 development eligible hours
* **Pearson Correlation coefficient ($r$):** **0.8931**
* **Spearman Rank Correlation ($\rho$):** **0.8999**
* **Coefficient of Determination ($R^2$):** **0.7976**
* **OLS Linear Calibration Fit:**
  $$\widehat{\text{CO(GT)}} = 0.00586 \times \text{PT08.S1(CO)} - 4.328$$

**Temperature Residual Analysis:**
Evaluating OLS prediction residuals ($e = y - \hat{y}$) across temperature bands reveals:
* **Cold hours ($T < 10^\circ\text{C}$, $n = 581$):** Mean residual is **$+0.181\text{ mg/m}^3$** (observations sit modestly above the linear fit).
* **Warm hours ($T > 25^\circ\text{C}$, $n = 1,527$):** Mean residual is **$-0.015\text{ mg/m}^3$** (observations sit very close to the linear fit).

This asymmetry indicates that ambient temperature influences sensor response, supporting the hypothesis that temperature compensation may improve calibration accuracy.

Visualized in: `../reports/figures/eda/03_target_sensor_relationship_dev.png`.

### 6.2 Temporal Stability (Rolling Correlation with Explicit Sample Thresholds)

To prevent small-sample artifacts at the beginning of the timeline, rolling Pearson correlations were computed with explicit minimum observation thresholds (`min_periods`):
* **14-day Rolling Window:** `min_periods = 168` eligible hours (minimum 7 full days of observations).
* **30-day Rolling Window:** `min_periods = 360` eligible hours (minimum 15 full days of observations).

| Metric | 14-day Window (`min_periods=168`) | 30-day Window (`min_periods=360`) |
|:---|:---:|:---:|
| **Valid Windows ($N$)** | 4,936 | 5,217 |
| **Mean $r$** | **0.914** | **0.905** |
| **Median $r$** | 0.924 | 0.907 |
| **Std Dev** | 0.033 | 0.033 |
| **Minimum $r$** | **0.801** (2004-07-26, late July) | **0.800** (2004-08-11, mid-August) |
| **Maximum $r$** | 0.973 | 0.951 |
| **November 2004 Minimum $r$** | 0.855 | 0.900 |

**Stability Observation:**
With small-sample noise filtered out, the correlation between `PT08.S1(CO)` and `CO(GT)` remains strong ($r \ge 0.80$) across the entire development period, with an overall mean above 0.90. The modest dip observed in mid-summer coincides with lower overall CO concentrations during that period.

Visualized in: `../reports/figures/eda/04_rolling_correlation_stability.png`.

---

## 7. Descriptive Sensor Shifts: Early vs. Late Period

We compared sensor and environmental distributions between two distinct windows:
* **Early Development Window:** `2004-03-11` to `2004-05-11` (1,488 grid hours; 1,461 hours with all 5 sensors present).
* **Late Development Window:** `2004-11-15` to `2005-01-16` (1,512 grid hours; 1,380 hours with all 5 sensors present).

### 7.1 Measured Means and Percentage Shifts

| Column | Role / Target Gas | Early Mean | Late Mean | Absolute Shift | Percentage Shift |
|:---|:---|:---:|:---:|:---:|:---:|
| `PT08.S1(CO)` | CO-targeted sensor | 1,168.20 | 1,137.34 | -30.86 | **-2.64%** |
| `PT08.S2(NMHC)` | NMHC-targeted sensor | 943.07 | 962.30 | +19.23 | **+2.04%** |
| `PT08.S3(NOx)` | NOx-targeted sensor | 963.66 | 801.93 | -161.73 | **-16.78%** |
| `PT08.S4(NO2)` | NO2-targeted sensor | 1,606.42 | 1,266.17 | -340.25 | **-21.18%** |
| `PT08.S5(O3)` | O3-targeted sensor | 994.87 | 1,176.24 | +181.36 | **+18.23%** |
| `T` | Temperature (°C) | 16.09 | 11.03 | -5.06 | **-31.45%** |
| `AH` | Absolute Humidity | 0.902 | 0.788 | -0.114 | **-12.64%** |

### 7.2 Crucial Scientific Distinction: Drift vs. Seasonality

> [!WARNING]
> **Sensor drift cannot be claimed as proven.**  
> Although `PT08.S3(NOx)` and `PT08.S4(NO2)` exhibit large downward shifts (-16.8% and -21.2%) while `PT08.S5(O3)` shifts upward (+18.2%), these temporal differences **cannot be attributed solely to sensor aging or drift**.
>
> 1. **Seasonal Confounding:** Ambient temperature dropped by $5.06^\circ\text{C}$ (from $16.09^\circ\text{C}$ to $11.03^\circ\text{C}$), and absolute humidity dropped from $0.902$ to $0.788$. Metal-oxide sensor conductivity depends nonlinearly on atmospheric temperature and moisture.
> 2. **Atmospheric Chemistry:** Concentrations of ozone and nitrogen oxides naturally exhibit seasonal shifts.
>
> Downstream models must evaluate whether environmental compensation accounts for these shifts before concluding that physical sensor drift has occurred.

Visualized in: `../reports/figures/eda/05_sensor_distribution_early_vs_late.png`.

---

## 8. Environmental Patterns & Diurnal Cycles

* **Seasonal Progression:** Temperature ranges from $-1.9^\circ\text{C}$ to $44.6^\circ\text{C}$, exhibiting large annual seasonality.
* **Diurnal Rush Hour Peaks:** Reference `CO(GT)` concentration shows clear bimodal diurnal peaks:
  * Morning rush period: 08:00 – 09:00.
  * Evening rush period: 19:00 – 20:00.
  * Nighttime minimum: 03:00 – 05:00.

Visualized in: `../reports/figures/eda/06_environmental_seasonal_patterns.png`.

---

## 9. Recommendations and Ideas to Test for Downstream Issues

### 9.1 For Feature Engineering (Issue #3 — ZiyadAhmedAmmar)
*Ideas to test (ownership remains with Issue #3):*
1. **Handling Missing Outages per Contract:**
   Per `docs/data_contract.md` §10 rule 2: *"A lag that falls inside a missing period must be NaN, not the last value seen."* Forward-filling is prohibited. An idea to explore is adding binary missingness indicator features (e.g. `is_recent_outage`) alongside valid lags.
2. **Sensor-Temperature Interaction Features:**
   Given that cold temperatures shift observations above the linear fit, testing multiplicative interaction features ($\text{PT08.S1} \times T$, $\text{PT08.S1} \times \text{AH}$) is recommended.
3. **Cyclical Temporal Encodings:**
   The bimodal diurnal rush-hour pattern suggests testing cyclical representations ($\sin / \cos$) of hour-of-day.

### 9.2 For Modeling & Validation (Issue #4 — Mohammed-Ahmed-Alansi)
*Ideas to test (ownership remains with Issue #4):*
1. **Target Distribution Considerations:**
   The positive skewness (+1.36) and extreme peak observations (up to 11.9 mg/m³) mean squared-error losses may be sensitive to high-concentration events. Note that while **MAE** is the competition evaluation metric, loss functions (such as Huber loss or log-transforms) could be explored as modeling choices.
2. **Multi-Sensor Cross-Compensation:**
   Evaluating whether multi-sensor models (Experiment C) outperform single-sensor models (Experiment B) by learning cross-gas corrections.
3. **Expanding-Window Protocol:**
   Utilize the 3-fold expanding cross-validation protocol pinned in contract v1.1 (`dev_folds()`).

---

## 10. Audit Artifacts & Relative Figure Registry

All figures conform to publication standards (300 DPI) and are stored under relative paths from `docs/`:

1. [Figure 1: Missingness Timeline](../reports/figures/eda/01_missingness_timeline.png) — Binary missingness matrix across all 13 columns annotating outage blocks and `TEST_START`.
2. [Figure 2: Target Distribution & Monthly Boxplots](../reports/figures/eda/02_target_distribution_and_monthly.png) — `CO(GT)` distribution (KDE) and monthly boxplots in the development period.
3. [Figure 3: Sensor-Target Relationship & Temperature Modulation](../reports/figures/eda/03_target_sensor_relationship_dev.png) — Scatter plot with OLS linear fit ($r = 0.8931$) and temperature gradient.
4. [Figure 4: Rolling Correlation Stability](../reports/figures/eda/04_rolling_correlation_stability.png) — 14-day and 30-day rolling Pearson correlation with `min_periods` thresholds (168h and 360h).
5. [Figure 5: Descriptive Sensor Distributions (Early vs. Late)](../reports/figures/eda/05_sensor_distribution_early_vs_late.png) — Comparative distribution curves for all 5 low-cost sensors (Spring '04 vs Winter '04-'05).
6. [Figure 6: Environmental Trends & Diurnal Cycles](../reports/figures/eda/06_environmental_seasonal_patterns.png) — Seasonal temperature/humidity trends and diurnal roadside profiles.
