from __future__ import annotations

import pandas as pd

from thesis.data.inventory import metadata_for_column, variable_inventory
from thesis.tables.common import (
    DATE_COLUMN,
    EXPECTATION_ADJUSTED_AIS,
    RAW_AIS,
    ROBUST_EXPECTATION_ADJUSTED_AIS,
)

def table_01_data_inventory(df: pd.DataFrame) -> pd.DataFrame:
    inventory = variable_inventory(df)
    if "role" not in inventory.columns and "expected_sign_or_role" in inventory.columns:
        inventory["role"] = inventory["expected_sign_or_role"]
    if "frequency" not in inventory.columns and "transformed_frequency" in inventory.columns:
        inventory["frequency"] = inventory["transformed_frequency"]
    if "observations" not in inventory.columns and "final_usable_observations" in inventory.columns:
        inventory["observations"] = inventory["final_usable_observations"]
    if "missing" not in inventory.columns and "missing_days" in inventory.columns:
        inventory["missing"] = inventory["missing_days"]
    if "missing_pct" not in inventory.columns:
        numeric_missing = pd.to_numeric(inventory["missing"], errors="coerce")
        inventory["missing_pct"] = numeric_missing / len(df)
    for column in ["units", "notes"]:
        if column not in inventory.columns:
            inventory[column] = inventory["variable"].map(
                _units_for_variable if column == "units" else _notes_for_variable
            )
    return inventory[
        [
            "variable",
            "source",
            "role",
            "date_range",
            "frequency",
            "units",
            "transformation",
            "observations",
            "missing",
            "missing_pct",
            "notes",
        ]
    ]

def _units_for_variable(column: str) -> str:
    meta = metadata_for_column(column)
    if hasattr(meta, "units"):
        return meta.units
    if column.endswith("_log_return"):
        return "Log points"
    if "sentiment" in column or column.endswith("_ais"):
        return "Sentiment index units"
    if "vix" in column or column in {"epu", "gpr"}:
        return "Index points"
    if "kalshi" in column:
        return "Price/probability points"
    if "metaculus" in column:
        return "Days"
    if column == DATE_COLUMN:
        return "Date"
    return "Price or source units"

def _notes_for_variable(column: str) -> str:
    if column == RAW_AIS:
        return "One-sided filtered one-factor DFM score from standardized GDELT/news and Reddit/social-media sentiment"
    if column == EXPECTATION_ADJUSTED_AIS:
        return "Residual from baseline expectation orthogonalization"
    if column == ROBUST_EXPECTATION_ADJUSTED_AIS:
        return "Residual from robustness orthogonalization including macro-risk controls"
    meta = metadata_for_column(column)
    if hasattr(meta, "notes") and meta.notes:
        return meta.notes
    if hasattr(meta, "expected_sign_or_role") and meta.expected_sign_or_role != "Unclassified variable":
        return meta.expected_sign_or_role
    return ""
