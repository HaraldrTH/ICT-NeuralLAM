"""Persistence for tidy evaluation tables and run metadata."""

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def write_results(records: Iterable[Dict[str, object]], path, file_format="csv"):
    """Write metric records to CSV or Parquet and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("Cannot write an empty evaluation result set")
    if file_format == "csv":
        frame.to_csv(output_path, index=False)
    elif file_format == "parquet":
        try:
            frame.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Parquet output requires a pandas-compatible engine such as "
                "pyarrow or fastparquet"
            ) from exc
    else:
        raise ValueError("file_format must be 'csv' or 'parquet'")
    return output_path


def write_run_metadata(metadata: Dict[str, object], path):
    """Write config/checkpoint provenance as formatted JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True, default=str)
    return output_path
