#!/usr/bin/env python3
"""run_all.py — runs the full Builder-B analysis pipeline in dependency
order:

  1. sql/*.sql            -> reports/tables/*.csv           (run_sql)
  2. customer_analytics.py -> activation/churn/persona CSVs (needs raw DB only)
  3. llm_analytics.py      -> model/task CSVs                (needs raw DB only)
  4. pricing_analytics.py  -> margin/A-B-test/scenario CSVs  (needs raw DB only)
  5. make_figures.py       -> reports/figures/*.png          (needs steps 1-4's CSVs)
  6. build_insights.py     -> reports/insights.md            (needs steps 1-4's CSVs)

Equivalent to `make analysis`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import build_insights, customer_analytics, llm_analytics, make_figures, pricing_analytics, run_sql  # noqa: E402


def main() -> None:
    start = time.time()

    print("=" * 70)
    print("1/6  SQL queries -> reports/tables/*.csv")
    print("=" * 70)
    run_sql.main()

    print("\n" + "=" * 70)
    print("2/6  Customer analytics (activation, churn model, personas)")
    print("=" * 70)
    customer_analytics.run_all()

    print("\n" + "=" * 70)
    print("3/6  LLM analytics (model x task, routing policy)")
    print("=" * 70)
    llm_analytics.run_all()

    print("\n" + "=" * 70)
    print("4/6  Pricing analytics (unit economics, A/B test, scenarios)")
    print("=" * 70)
    pricing_analytics.run_all()

    print("\n" + "=" * 70)
    print("5/6  Figures -> reports/figures/*.png")
    print("=" * 70)
    make_figures.run_all()

    print("\n" + "=" * 70)
    print("6/6  Insights report -> reports/insights.md")
    print("=" * 70)
    build_insights.run_all()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
