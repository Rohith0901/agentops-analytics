"""db.py — tiny shared helper for talking to the AgentOps SQLite database.

Every other module (analysis, dashboard, Excel export) imports `ROOT`, `DB_PATH`,
`get_conn()` and `query()` from here instead of re-deriving paths or opening
connections themselves, so there is exactly one place that knows where the
database lives.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

# Project root, resolved relative to this file (works no matter the caller's cwd).
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "agentops.db"


def get_conn() -> sqlite3.Connection:
    """Open a new SQLite connection to data/agentops.db.

    Callers are responsible for closing the connection (or use it as a
    context manager). A fresh connection per call keeps this safe to use
    from multiple threads (e.g. a Streamlit dashboard).
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} does not exist yet — run `python src/generate_data.py` first."
        )
    return sqlite3.connect(DB_PATH)


def query(sql: str, params: dict | tuple | None = None) -> pd.DataFrame:
    """Run a single SQL statement against the database and return a DataFrame."""
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def _strip_sql_comments(sql_text: str) -> str:
    """Remove `--` line comments and `/* ... */` block comments from SQL text.

    Done carefully so that `--` or `/*` inside a quoted string literal is not
    mistaken for a comment marker.
    """
    out = []
    i = 0
    n = len(sql_text)
    in_single = False
    in_double = False
    while i < n:
        ch = sql_text[i]
        nxt = sql_text[i + 1] if i + 1 < n else ""

        if in_single:
            out.append(ch)
            if ch == "'" and nxt != "'":
                in_single = False
            i += 1
            continue
        if in_double:
            out.append(ch)
            if ch == '"' and nxt != '"':
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            out.append(ch)
            i += 1
            continue

        if ch == "-" and nxt == "-":
            # Line comment: skip to end of line (keep the newline itself).
            j = sql_text.find("\n", i)
            i = j if j != -1 else n
            continue
        if ch == "/" and nxt == "*":
            # Block comment: skip to closing */
            j = sql_text.find("*/", i + 2)
            i = j + 2 if j != -1 else n
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def run_sql_file(path: str | Path) -> pd.DataFrame:
    """Execute a .sql file that may contain several statements.

    Only the *final* statement is expected to be a SELECT; earlier statements
    (e.g. temp view/table setup) are executed for their side effects and the
    result of the last SELECT is returned as a DataFrame. Comments are
    stripped first so that `--` notes don't break statement splitting.
    """
    path = Path(path)
    raw = path.read_text()
    cleaned = _strip_sql_comments(raw)

    # Split on semicolons that terminate a statement; drop empty fragments.
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if not statements:
        raise ValueError(f"{path} contains no SQL statements.")

    with get_conn() as conn:
        for stmt in statements[:-1]:
            conn.execute(stmt)
        return pd.read_sql_query(statements[-1], conn)
