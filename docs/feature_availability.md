# Feature Availability & Leakage Assessment

**Project:** Time-Aware Air Quality Sensor Calibration: CO Estimation and Drift Evaluation  
**Issue:** #3 — Feature Engineering & Leakage Tests  
**Author:** Emran Yaseen Al-Sabry (`EmranYaseen-ALSabry`)  
**Status:** Complete  
**Date:** 2026-10-24  
**Data Contract Version:** v1.1 (`docs/data_contract.md`)  

---

## 1. Overview & Interface Specification

This document provides a complete registry and causal leakage audit for all engineered features produced by `src/air_quality/features.py`.

In strict adherence to `docs/data_contract.md` §10:
* Features are constructed on the **complete hourly grid** returned by `air_quality.data.load_raw()` before any row filtering.
* Features use only information available at declared prediction time: current and past low-cost device readings, ambient environmental variables, and timestamps.
* All temporal windows look **strictly backwards**.
* **Zero Target & Deployment Leakage:** Neither current nor past `CO(GT)`, nor reference gas measurements (`FORBIDDEN_COLS`: `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`), are ever used in feature creation.

### 1.1 Public API Interface

```python
from air_quality.features import build_features

# Default: builds required groups ("time", "lags")
features_default = build_features(df)

# Specific groups for ablation experiments in Issue #4:
features_time = build_features(df, groups="time")
features_lags = build_features(df, groups="lags")
features_custom = build_features(df, groups=["time", "env_interaction"])

# All available feature groups (30 features):
features_all = build_features(df, groups="all")
```

---

## 2. Causal Temporal Integrity & Look-Back Boundary

### 2.1 The Backwards-Looking Rule
Every engineered feature at observation time $t$ uses information from time $s \le t$ only:
* Instantaneous features (e.g. `time`, `env_interaction`) use inputs at time $t$.
* Lagged features use inputs at $t - 1\text{ h}$, $t - 3\text{ h}$, and $t - 24\text{ h}$.
* Rolling window features apply `shift(1)` **prior** to rolling, aggregating inputs from $[t - W, t - 1\text{ h}]$ and strictly excluding current hour $t$.

### 2.2 Maximum Look-Back Window: 24 Hours
$$\text{LONGEST\_LOOKBACK\_HOURS} = 24$$

The longest backwards reach across all feature groups is exactly **24 hours** (from 24h lags and 24h past rolling statistics).

### 2.3 Justification for Zero Cross-Validation Gap (`GAP_HOURS = 0`)
In `docs/data_contract.md` §9, cross-validation folds are configured with `GAP_HOURS = 0`. This decision is scientifically sound because:
1. Since features reach back at most 24 hours and never look forward ($s \le t$), training folds cannot access future validation period information.
2. When evaluating at validation hour $t_\text{val}$, reaching back into late training data represents the exact operational setting of a deployed continuous monitoring model.
3. Because no feature uses the target `CO(GT)`, no ground-truth label information crosses fold boundaries.

---

## 3. Comprehensive Feature Registry

A total of **30 engineered features** are provided across 5 modular groups:

| Feature Name | Group | Source Column(s) | Look-back | Physical & Data-Driven Motivation |
|:---|:---:|:---:|:---:|:---|
| `hour_sin` | `time` | `timestamp.hour` | 0 h | Continuous cyclical representation of 24h diurnal cycle; aligns hour 23 with hour 0. |
| `hour_cos` | `time` | `timestamp.hour` | 0 h | Orthogonal component of diurnal cyclical encoding. |
| `dow_sin` | `time` | `timestamp.dayofweek` | 0 h | Continuous cyclical encoding of weekly traffic and commercial emission cycles (7 days). |
| `dow_cos` | `time` | `timestamp.dayofweek` | 0 h | Orthogonal weekly cyclical component. |
| `dayofweek` | `time` | `timestamp.dayofweek` | 0 h | Linear integer representation of day of week (0 = Monday, 6 = Sunday). |
| `is_weekend` | `time` | `timestamp.dayofweek` | 0 h | Binary indicator (1.0 for Saturday/Sunday) capturing weekend reductions in heavy traffic. |
| `PT08.S1(CO)_lag_1h` | `lags` | `PT08.S1(CO)` | 1 h | Short-term autoregressive sensor inertia and local concentration persistence. |
| `PT08.S2(NMHC)_lag_1h` | `lags` | `PT08.S2(NMHC)` | 1 h | 1h lag for NMHC-targeted sensor. |
| `PT08.S3(NOx)_lag_1h` | `lags` | `PT08.S3(NOx)` | 1 h | 1h lag for NOx-targeted sensor. |
| `PT08.S4(NO2)_lag_1h` | `lags` | `PT08.S4(NO2)` | 1 h | 1h lag for NO2-targeted sensor. |
| `PT08.S5(O3)_lag_1h` | `lags` | `PT08.S5(O3)` | 1 h | 1h lag for O3-targeted sensor. |
| `PT08.S1(CO)_lag_3h` | `lags` | `PT08.S1(CO)` | 3 h | Intermediate-term temporal trend capturing multi-hour traffic rush dynamics. |
| `PT08.S2(NMHC)_lag_3h` | `lags` | `PT08.S2(NMHC)` | 3 h | 3h lag for NMHC sensor. |
| `PT08.S3(NOx)_lag_3h` | `lags` | `PT08.S3(NOx)` | 3 h | 3h lag for NOx sensor. |
| `PT08.S4(NO2)_lag_3h` | `lags` | `PT08.S4(NO2)` | 3 h | 3h lag for NO2 sensor. |
| `PT08.S5(O3)_lag_3h` | `lags` | `PT08.S5(O3)` | 3 h | 3h lag for O3 sensor. |
| `PT08.S1(CO)_lag_24h` | `lags` | `PT08.S1(CO)` | 24 h | Daily seasonal baseline; captures yesterday's reading at the exact same hour. |
| `PT08.S2(NMHC)_lag_24h` | `lags` | `PT08.S2(NMHC)` | 24 h | 24h daily lag for NMHC sensor. |
| `PT08.S3(NOx)_lag_24h` | `lags` | `PT08.S3(NOx)` | 24 h | 24h daily lag for NOx sensor. |
| `PT08.S4(NO2)_lag_24h` | `lags` | `PT08.S4(NO2)` | 24 h | 24h daily lag for NO2 sensor. |
| `PT08.S5(O3)_lag_24h` | `lags` | `PT08.S5(O3)` | 24 h | 24h daily lag for O3 sensor. |
| `PT08.S1(CO)_roll_mean_6h` | `rolling` | `PT08.S1(CO)` | 6 h (past-only) | 6-hour moving average of CO sensor baseline (excluding current hour $t$). |
| `PT08.S1(CO)_roll_std_6h` | `rolling` | `PT08.S1(CO)` | 6 h (past-only) | 6-hour rolling volatility / turbulence indicator. |
| `PT08.S1(CO)_roll_mean_24h`| `rolling` | `PT08.S1(CO)` | 24 h (past-only) | 24-hour background level smoothing out high-frequency noise. |
| `PT08.S1(CO)_roll_std_24h` | `rolling` | `PT08.S1(CO)` | 24 h (past-only) | 24-hour diurnal dispersion amplitude. |
| `T_x_RH` | `env_interaction` | `T`, `RH` | 0 h | Interaction between temperature and relative humidity representing dew-point / psychrometric state. |
| `PT08.S1_x_T` | `env_interaction` | `PT08.S1(CO)`, `T` | 0 h | Primary tin-oxide temperature compensation term motivated by EDA residual findings. |
| `PT08.S1_x_AH` | `env_interaction` | `PT08.S1(CO)`, `AH`| 0 h | Tin-oxide absolute humidity cross-sensitivity term. |
| `inputs_missing_lag_1h` | `missing_ind` | `inputs_available` | 1 h | Binary flag indicating if low-cost inputs were unobserved at $t-1\text{ h}$. |
| `inputs_missing_lag_24h` | `missing_ind` | `inputs_available` | 24 h | Binary flag indicating if low-cost inputs were unobserved at $t-24\text{ h}$. |

---

## 4. Leakage Assessment & Verification Suite

All features were systematically subjected to automated validation tests in `tests/test_features.py`:

### 4.1 Temporal Causal Isolation (Future-to-Past)
* **Test:** `test_no_temporal_leakage_future_to_past`
* **Mechanism:** In an evaluation timeline with boundary $t$, all input values at $t + k$ ($k \ge 1$) were violently perturbed with extreme values ($> 9000$).
* **Result:** Every feature output for all rows $\le t$ was verified to be strictly identical (`pd.testing.assert_frame_equal`). No future information enters past or current calculations.

### 4.2 Past-Only Rolling Verification
* **Test:** `test_past_only_rolling_excludes_current_hour`
* **Mechanism:** For time $t$, perturbing input values at current hour $t$ produces zero change in `roll_mean_6h` or `roll_std_6h` at time $t$. The effect appears only at $t+1$.
* **Result:** Confirms that `shift(1)` is properly executed before rolling windows.

### 4.3 Target and Deployment Leakage Immunity
* **Test:** `test_no_target_or_deployment_leakage`
* **Mechanism:** Injected random noise into `CO(GT)` and all four reference gases (`NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`).
* **Result:** Feature outputs were completely unchanged. Features depend purely on permissible device inputs.

### 4.4 Hourly Grid Gap Integrity (No Silent Bridging)
* **Test:** `test_gap_integrity_lags_do_not_bridge_gaps`
* **Mechanism:** Tested on an artificial non-consecutive grid where hour 12:00 is missing between 11:00 and 13:00.
* **Result:** The 1-hour lag at 13:00 produces **`NaN`** (since 12:00 was unobserved). It does NOT bridge the gap to return the 11:00 observation.

---

## 5. Downstream Integration Guidance (Issue #4 — Mohammed-Ahmed-Alansi)

Issue #4 executes the ablation experiments:
* **Baseline C:** Ridge on `INPUT_COLS` (`PT08.S1` to `PT08.S5`, `T`, `RH`, `AH`).
* **Experiment D:** Baseline C + selected engineered feature groups.
  - To test each group independently under the pre-registered selection rule:
    ```python
    # Example ablation evaluation:
    for group in ["time", "lags", "env_interaction", "rolling", "missing_ind"]:
        X_feats = build_features(df, groups=group)
        X_combined = pd.concat([df[INPUT_COLS], X_feats], axis=1)
        # Evaluate under primary 3-fold dev_folds() protocol
    ```
* **Experiment E:** Best justified feature combination with `HistGradientBoostingRegressor`.
