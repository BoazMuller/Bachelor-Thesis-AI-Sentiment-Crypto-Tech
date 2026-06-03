from types import SimpleNamespace

import pandas as pd
import pytest

from thesis.table_output import write_registered_table
from thesis.tables.common import (
    EXPECTATION_ADJUSTED_AIS,
    LAGGED_EXPECTATION_ADJUSTED_AIS,
    RAW_AIS,
    expectation_adjusted_sample,
    raw_ais_sample,
)
from thesis.tables.data_inventory import table_01_data_inventory


def test_write_registered_table_rejects_non_csv_formats(monkeypatch, tmp_path):
    spec = SimpleNamespace(output_dir=tmp_path, csv_filename="table.csv")
    monkeypatch.setattr("thesis.table_output.get_table_spec", lambda _: spec)

    with pytest.raises(ValueError, match="Only CSV"):
        write_registered_table(1, pd.DataFrame({"x": [1]}), formats=["tex"])


def test_write_registered_table_writes_only_csv(monkeypatch, tmp_path):
    spec = SimpleNamespace(output_dir=tmp_path, csv_filename="table.csv")
    monkeypatch.setattr("thesis.table_output.get_table_spec", lambda _: spec)

    outputs = write_registered_table(1, pd.DataFrame({"x": [1]}))

    assert outputs == [tmp_path / "table.csv"]
    assert (tmp_path / "table.csv").exists()
    assert not list(tmp_path.glob("*.tex"))
    assert not list(tmp_path.glob("*.md"))


def test_eais_sample_filtering_keeps_raw_ais_fuller_sample():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="D"),
            RAW_AIS: [0.1, 0.2, 0.3, 0.4],
            EXPECTATION_ADJUSTED_AIS: [pd.NA, pd.NA, 0.0, 0.1],
            LAGGED_EXPECTATION_ADJUSTED_AIS: [pd.NA, pd.NA, pd.NA, 0.0],
        }
    )

    eais = expectation_adjusted_sample(df)
    lagged_eais = expectation_adjusted_sample(df, include_lagged=True)
    raw = raw_ais_sample(df)

    assert eais["date"].min() == pd.Timestamp("2025-01-03")
    assert lagged_eais["date"].min() == pd.Timestamp("2025-01-04")
    assert raw["date"].min() == pd.Timestamp("2025-01-01")


def test_data_inventory_includes_model_usage_column_for_derived_variables():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="D"),
            RAW_AIS: [0.1, 0.2, 0.3],
            EXPECTATION_ADJUSTED_AIS: [pd.NA, 0.0, 0.1],
            LAGGED_EXPECTATION_ADJUSTED_AIS: [pd.NA, pd.NA, 0.0],
            "bitcoin_conditional_volatility": [0.2, 0.3, 0.4],
        }
    )

    inventory = table_01_data_inventory(df)

    assert "used_in_models" in inventory.columns
    raw_usage = inventory.loc[inventory["variable"].eq(RAW_AIS), "used_in_models"].iloc[0]
    vol_usage = inventory.loc[
        inventory["variable"].eq("bitcoin_conditional_volatility"),
        "used_in_models",
    ].iloc[0]
    assert "dfm_em" in raw_usage
    assert "tvpvar_connectedness" in vol_usage
