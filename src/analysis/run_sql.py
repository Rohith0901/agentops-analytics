#!/usr/bin/env python3
"""run_sql.py — execute every analytical query in sql/*.sql against the
AgentOps database and write each result to reports/tables/<name>.csv.

Security note: every file in sql/ is a project-authored, hard-coded query
(never built from user input), and is executed as-is via db.run_sql_file,
which itself uses no string interpolation of external data — see
db.py::run_sql_file. This is the one place allowed to "run arbitrary SQL
files" because the set of files is fixed at development time, not chosen by
a runtime input.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/db.py` importable whether this script is run as
# `python src/analysis/run_sql.py` or as part of the `src.analysis` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import ROOT, run_sql_file  # noqa: E402

SQL_DIR = ROOT / "sql"
TABLES_DIR = ROOT / "reports" / "tables"


def run_all_sql(sql_dir: Path = SQL_DIR, out_dir: Path = TABLES_DIR) -> dict[str, int]:
    """Run every .sql file in `sql_dir` and write results to `out_dir`.

    Returns a dict mapping each query name to its output row count, for
    logging / testing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}

    sql_files = sorted(sql_dir.glob("*.sql"))
    if not sql_files:
        raise FileNotFoundError(f"No .sql files found in {sql_dir}")

    for sql_path in sql_files:
        name = sql_path.stem
        df = run_sql_file(sql_path)
        if df.empty:
            raise ValueError(f"{sql_path.name} returned an empty result set.")
        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        row_counts[name] = len(df)
        print(f"  {name:<28s} -> {len(df):>6,d} rows -> {out_path.relative_to(ROOT)}")

    return row_counts


def main() -> None:
    print(f"Running {len(sorted(SQL_DIR.glob('*.sql')))} SQL files from {SQL_DIR} ...")
    row_counts = run_all_sql()
    print(f"Done. Wrote {len(row_counts)} tables to {TABLES_DIR}.")


if __name__ == "__main__":
    main()
