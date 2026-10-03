"""Print the shape and columns of tables, without loading big files into memory (read-only helper).

    python scripts/inspect_tables.py <file> [<file> ...]

For each CSV or Parquet file it prints the row count, every column with its type, and the first 3 rows. A table of
30 rows or fewer is printed whole, because small result tables are the evidence itself. A JSON file prints its top-level
keys. Nothing is written and no flight data is modified. Paste the output back when a script needs to know a schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SMALL_TABLE_ROWS = 30
HEAD_ROWS = 3


def csv_row_count(path: Path, chunk: int = 1_000_000) -> int:
    """Count data rows by streaming the first column, so a 3-million-row file needs little memory."""
    total = 0
    for part in pd.read_csv(path, usecols=[0], chunksize=chunk, low_memory=False):
        total += len(part)
    return total


def describe_csv(path: Path) -> tuple[int, pd.DataFrame]:
    rows = csv_row_count(path)
    sample = pd.read_csv(path, nrows=SMALL_TABLE_ROWS if rows <= SMALL_TABLE_ROWS else 10_000, low_memory=False)
    return rows, sample


def describe_parquet(path: Path) -> tuple[int, pd.DataFrame]:
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    rows = handle.metadata.num_rows
    batch = next(handle.iter_batches(batch_size=SMALL_TABLE_ROWS if rows <= SMALL_TABLE_ROWS else 10_000), None)
    return rows, (batch.to_pandas() if batch is not None else pd.DataFrame())


def show(path: Path) -> None:
    print("=" * 100)
    print(path)
    if not path.exists():
        print("  NOT FOUND")
        return
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"  JSON, top-level keys: {list(payload) if isinstance(payload, dict) else type(payload).__name__}")
        return
    if suffix == ".csv":
        rows, sample = describe_csv(path)
    elif suffix == ".parquet":
        rows, sample = describe_parquet(path)
    else:
        print(f"  not a csv, parquet or json file ({suffix})")
        return
    print(f"  rows: {rows:,}   columns: {sample.shape[1]}")
    for name, dtype in sample.dtypes.items():
        print(f"    {name}: {dtype}")
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 40):
        if rows <= SMALL_TABLE_ROWS:
            print("  whole table:")
            print(sample.to_string(index=False))
        else:
            print(f"  first {HEAD_ROWS} rows:")
            print(sample.head(HEAD_ROWS).to_string(index=False))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for item in argv:
        show(Path(item))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
