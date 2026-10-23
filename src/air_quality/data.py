"""Data acquisition, loading, and validation for the UCI Air Quality dataset.

The contract implemented here is documented in ``docs/data_contract.md``.
Run ``python -m air_quality.data`` to download (if needed), verify, and
print the validation report.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --- Source and integrity -------------------------------------------------

DATASET_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
DATASET_DOI = "10.24432/C59K5F"
ARCHIVE_SHA256 = "d4a64013fb385288a8a48d9d193ca7079b2e1bbddf6f8d458feb8c08ab2b8a2a"
CSV_NAME = "AirQualityUCI.csv"
CSV_SHA256 = "13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# --- Schema ---------------------------------------------------------------

MISSING_MARKER = -200
TIMESTAMP_FORMAT = "%d/%m/%Y %H.%M.%S"

TARGET = "CO(GT)"
SENSOR_COLS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
ENV_COLS = ["T", "RH", "AH"]
INPUT_COLS = SENSOR_COLS + ENV_COLS
# Reference-analyzer measurements: never valid model inputs (deployment leakage).
FORBIDDEN_COLS = ["NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
NUMERIC_COLS = [TARGET, *SENSOR_COLS, *ENV_COLS, *FORBIDDEN_COLS]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw_dir: Path | None = None, force: bool = False) -> Path:
    """Fetch the official archive, verify it, and extract the CSV.

    Returns the path of the verified CSV. An existing CSV with the expected
    hash is reused unless ``force`` is true. Any hash mismatch raises.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / CSV_NAME
    if csv_path.exists() and not force and sha256_of(csv_path) == CSV_SHA256:
        return csv_path

    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        payload = response.read()
    archive_hash = hashlib.sha256(payload).hexdigest()
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(f"Archive SHA-256 mismatch: got {archive_hash}, expected {ARCHIVE_SHA256}")
    (raw_dir / "air+quality.zip").write_bytes(payload)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_bytes = archive.read(CSV_NAME)
    if hashlib.sha256(csv_bytes).hexdigest() != CSV_SHA256:
        raise ValueError(f"{CSV_NAME} SHA-256 mismatch inside the archive")
    csv_path.write_bytes(csv_bytes)
    return csv_path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the CSV onto a complete hourly grid.

    - ``;`` separator, ``,`` decimal; blank trailing columns and rows dropped.
    - ``Date`` + ``Time`` parsed strictly into a ``timestamp`` index (naive
      local time as recorded; the source does not state a DST convention).
    - ``-200`` converted to NaN in every numeric column.
    - Rows with a missing target or missing inputs are kept: filtering is a
      separate step (see :func:`eligible_mask`), applied after the split.
    """
    path = Path(path) if path is not None else DEFAULT_RAW_DIR / CSV_NAME
    raw = pd.read_csv(path, sep=";", decimal=",", dtype={"Date": str, "Time": str})
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    raw = raw.dropna(how="all")

    missing_cols = sorted(set(NUMERIC_COLS + ["Date", "Time"]) - set(raw.columns))
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    timestamp = pd.to_datetime(raw["Date"] + " " + raw["Time"], format=TIMESTAMP_FORMAT)
    if timestamp.duplicated().any():
        raise ValueError(f"{int(timestamp.duplicated().sum())} duplicate timestamps")
    if ((timestamp.dt.minute != 0) | (timestamp.dt.second != 0)).any():
        raise ValueError("Timestamps not aligned to the hour")

    df = raw[NUMERIC_COLS].astype(float).replace(MISSING_MARKER, np.nan)
    df.index = pd.DatetimeIndex(timestamp, name="timestamp")
    df = df.sort_index()
    grid = pd.date_range(df.index[0], df.index[-1], freq="h", name="timestamp")
    return df.reindex(grid)


def inputs_available(df: pd.DataFrame) -> pd.Series:
    """True where at least one current low-cost input is present."""
    return df[INPUT_COLS].notna().any(axis=1)


def eligible_mask(df: pd.DataFrame) -> pd.Series:
    """Rows usable for training and primary evaluation.

    Eligible = valid target AND at least one current low-cost input present.
    Hours with no current input cannot produce an estimate at prediction time.
    """
    return df[TARGET].notna() & inputs_available(df)


def validate(df: pd.DataFrame) -> dict:
    """Summarise grid integrity, missingness, and eligibility counts."""
    has_inputs = inputs_available(df)
    target_ok = df[TARGET].notna()
    all_inputs_present = df[INPUT_COLS].notna().all(axis=1)
    full_grid = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "missing_grid_hours": int(len(full_grid.difference(df.index))),
        "missing_per_column": df[NUMERIC_COLS].isna().sum().to_dict(),
        "target_valid": int(target_ok.sum()),
        "target_missing": int((~target_ok).sum()),
        "hours_without_inputs": int((~has_inputs).sum()),
        "hours_without_inputs_with_target": int((~has_inputs & target_ok).sum()),
        "hours_partial_inputs": int((has_inputs & ~all_inputs_present).sum()),
        "eligible": int(eligible_mask(df).sum()),
    }


def main() -> None:
    csv_path = download()
    print(f"CSV: data/raw/{csv_path.name} (SHA-256 verified)")
    report = validate(load_raw(csv_path))
    missing = report.pop("missing_per_column")
    for key, value in report.items():
        print(f"{key:34s} {value}")
    print("missing per column:")
    for col, n in missing.items():
        print(f"  {col:14s} {n}")


if __name__ == "__main__":
    main()
