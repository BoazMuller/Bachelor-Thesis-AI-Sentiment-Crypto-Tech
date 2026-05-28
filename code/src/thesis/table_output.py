from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from thesis.table_registry import get_table_spec


def write_registered_table(
    table_number: int,
    df: pd.DataFrame,
    *,
    formats: Iterable[str] = ("csv", "tex"),
) -> list[Path]:
    spec = get_table_spec(table_number)
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    requested = set(formats)
    table = _format_dates(df)

    if "csv" in requested:
        path = spec.output_dir / spec.csv_filename
        table.to_csv(path, index=False)
        outputs.append(path)

    if "tex" in requested:
        path = spec.output_dir / spec.tex_filename
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                table.to_latex(
                    index=False,
                    escape=True,
                    caption=spec.title,
                    label=spec.latex_label,
                    float_format="%.6g",
                )
            )
        outputs.append(path)

    if "md" in requested:
        path = spec.output_dir / spec.md_filename
        try:
            path.write_text(table.to_markdown(index=False), encoding="utf-8")
        except ImportError:
            path.write_text(table.to_string(index=False), encoding="utf-8")
        outputs.append(path)

    return outputs


def write_dataframe(
    df: pd.DataFrame,
    path: Path,
    *,
    include_index: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _format_dates(df).to_csv(path, index=include_index)
    return path


def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    for column in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[column]):
            table[column] = table[column].dt.date
    return table
