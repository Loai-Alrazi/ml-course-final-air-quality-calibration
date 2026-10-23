# Problem Definition

## Task

Supervised regression: estimate the hourly reference carbon monoxide concentration **`CO(GT)`** (mg/m³), as measured by the co-located certified reference analyzer, from the low-cost multi-sensor device's readings for the same hour.

- **Observation unit:** one hour.
- **Prediction time:** after the current hour's low-cost sensor and environmental readings are available.
- **This is current-hour calibration, not forecasting.** Predicting the next hour or day would be a different problem, with a different target and different feature availability.

## Research question

> Can a multi-sensor calibration model, supported by justified feature engineering, estimate reference CO concentrations on future chronological data more accurately than a simple linear calibration of the CO-targeted sensor alone?

The question is answered by the controlled experiments A–E defined in `PROJECT_SPEC.md` §9, all evaluated under the same time-aware protocol (`docs/data_contract.md`). No improvement is assumed in advance.

## Inputs

**Allowed** — available from the low-cost device at prediction time:

| Column | Description |
|---|---|
| `PT08.S1(CO)` | Tin-oxide sensor response, nominally CO-targeted |
| `PT08.S2(NMHC)` | Titania sensor response, nominally NMHC-targeted |
| `PT08.S3(NOx)` | Tungsten-oxide sensor response, nominally NOx-targeted |
| `PT08.S4(NO2)` | Tungsten-oxide sensor response, nominally NO2-targeted |
| `PT08.S5(O3)` | Indium-oxide sensor response, nominally O3-targeted |
| `T`, `RH`, `AH` | Temperature (°C), relative humidity (%), absolute humidity |
| time features | Derived from the timestamp (hour, day of week, …) |

Engineered features may use current and **past** values of these inputs only.

**Not allowed:**

- `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)` — outputs of the reference analyzer. A low-cost deployment has no reference analyzer, so using them would be deployment leakage.
- Current or past `CO(GT)` — the quantity being estimated; not available to a low-cost-only system.

## Evaluation summary

- Primary metric: MAE (mg/m³). Secondary: RMSE, R².
- Chronological final test from `2005-01-17 00:00`, used once after all decisions are frozen.
- Time-aware cross-validation on the development period for every comparison.

## Beneficiaries and limitations

Potential beneficiaries are air-quality monitoring researchers and teams evaluating low-cost sensing systems.

This is an academic model and must not be presented as a certified replacement for regulatory monitoring instruments. The data come from one device at one Italian roadside location over about 13 months (2004–2005). Results do not establish performance at other sites, with other devices, or in other periods.
