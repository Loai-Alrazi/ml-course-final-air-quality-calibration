import numpy as np
import pandas as pd
import pytest

from air_quality import data
from air_quality.data import (
    DEFAULT_RAW_DIR,
    CSV_NAME,
    CSV_SHA256,
    FORBIDDEN_COLS,
    INPUT_COLS,
    TARGET,
    eligible_mask,
    load_raw,
    sha256_of,
    validate,
)

HEADER = ("Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC);NOx(GT);"
          "PT08.S3(NOx);NO2(GT);PT08.S4(NO2);PT08.S5(O3);T;RH;AH;;")
BLANK = ";" * 16


def row(date, time, co="2,6", sensors="1360", env=("13,6", "48,9", "0,7578")):
    # Column order follows the real file header.
    values = [date, time, co, sensors, "150", "11,9", sensors, "166", sensors, "113",
              sensors, sensors, *env]
    return ";".join(values) + ";;"


def write_csv(tmp_path, lines):
    path = tmp_path / "sample.csv"
    path.write_bytes(("\r\n".join([HEADER, *lines, BLANK, BLANK]) + "\r\n").encode("ascii"))
    return path


@pytest.fixture
def sample(tmp_path):
    lines = [
        row("10/03/2004", "18.00.00"),
        row("10/03/2004", "20.00.00", co="-200"),                                  # out of order
        row("10/03/2004", "19.00.00", sensors="-200", env=("-200", "-200", "-200")),
        row("10/03/2004", "21.00.00", co="-200", sensors="-200", env=("-200", "-200", "-200")),
        row("10/03/2004", "22.00.00", co="1,1"),
    ]
    return write_csv(tmp_path, lines)


def test_load_parses_format_and_drops_blank_rows_and_columns(sample):
    df = load_raw(sample)
    assert len(df) == 5
    assert not any(c.startswith("Unnamed") for c in df.columns)
    assert {"Date", "Time"}.isdisjoint(df.columns)
    assert df.loc["2004-03-10 18:00", TARGET] == pytest.approx(2.6)
    assert df.loc["2004-03-10 18:00", "AH"] == pytest.approx(0.7578)
    assert all(df[c].dtype == float for c in df.columns)


def test_missing_marker_becomes_nan(sample):
    df = load_raw(sample)
    assert not (df == -200).any().any()
    assert np.isnan(df.loc["2004-03-10 20:00", TARGET])
    assert df.loc["2004-03-10 19:00", INPUT_COLS].isna().all()


def test_index_is_sorted_complete_hourly_grid(sample):
    df = load_raw(sample)
    assert df.index.name == "timestamp"
    assert df.index.is_monotonic_increasing
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_missing_hours_are_inserted_as_empty_rows(tmp_path):
    path = write_csv(tmp_path, [row("10/03/2004", "18.00.00"), row("10/03/2004", "21.00.00")])
    df = load_raw(path)
    assert len(df) == 4
    assert df.loc["2004-03-10 19:00":"2004-03-10 20:00"].isna().all().all()


def test_duplicate_timestamps_raise(tmp_path):
    path = write_csv(tmp_path, [row("10/03/2004", "18.00.00"), row("10/03/2004", "18.00.00")])
    with pytest.raises(ValueError, match="duplicate"):
        load_raw(path)


def test_eligibility_requires_target_and_some_input(sample):
    df = load_raw(sample)
    expected = {
        "2004-03-10 18:00": True,    # target + inputs
        "2004-03-10 19:00": False,   # target, no inputs
        "2004-03-10 20:00": False,   # inputs, no target
        "2004-03-10 21:00": False,   # neither
        "2004-03-10 22:00": True,
    }
    mask = eligible_mask(df)
    for ts, ok in expected.items():
        assert bool(mask.loc[ts]) is ok, ts


def test_validate_counts(sample):
    report = validate(load_raw(sample))
    assert report["rows"] == 5
    assert report["target_valid"] == 3
    assert report["hours_without_inputs"] == 2
    assert report["hours_without_inputs_with_target"] == 1
    assert report["eligible"] == 2


def test_contract_column_groups_are_disjoint():
    assert TARGET not in INPUT_COLS
    assert set(FORBIDDEN_COLS).isdisjoint(INPUT_COLS)


def test_sha256_of(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert sha256_of(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


REAL_CSV = DEFAULT_RAW_DIR / CSV_NAME
requires_real = pytest.mark.skipif(not REAL_CSV.exists(), reason="raw CSV not downloaded")


@requires_real
def test_real_file_matches_documented_contract():
    assert sha256_of(REAL_CSV) == CSV_SHA256
    report = validate(load_raw(REAL_CSV))
    assert report["rows"] == 9357
    assert report["start"] == pd.Timestamp("2004-03-10 18:00")
    assert report["end"] == pd.Timestamp("2005-04-04 14:00")
    assert report["duplicate_timestamps"] == 0
    assert report["missing_grid_hours"] == 0
    assert report["target_valid"] == 7674
    assert report["hours_without_inputs"] == 366
    assert report["hours_without_inputs_with_target"] == 330
    assert report["hours_partial_inputs"] == 0
    assert report["eligible"] == 7344


def test_module_exposes_doi():
    assert data.DATASET_DOI == "10.24432/C59K5F"
