#!/usr/bin/env python
"""
analyze_benchmark.py — turns run_benchmark.py's results CSV into statistics and a
chart: pass@1 / pass@k, tokens/latency/cost, Wilson confidence intervals on pass
rates, and a chi-square test of model vs. outcome.

Usage
-----
    .venv/bin/python llm_benchmark/analyze_benchmark.py \\
        --results llm_benchmark/results/results.csv \\
        --out-dir llm_benchmark/results

Outputs
-------
    llm_benchmark/results/summary.md               — all tables, human-readable
    llm_benchmark/results/pass_rate_by_model.png    — bar chart w/ Wilson 95% CI
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- stats


def wilson_ci(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    More reliable than the naive normal-approximation interval at small n or
    extreme proportions (both common here with repeats=3-5 per task)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, Codex paper):

        pass@k = 1 - C(n-c, k) / C(n, k)

    where n = samples drawn for a task, c = number of those that passed,
    k = number of attempts a "user" gets. Uses the numerically stable
    product form rather than raw combinatorics to avoid overflow.
    """
    if n - c < k:
        return 1.0
    # product_{i=0}^{k-1} (n-c-i) / (n-i)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def pass_at_k_grouped(df: pd.DataFrame, group_cols: list[str], k: int) -> pd.Series:
    """Compute mean pass@k across tasks, grouped by group_cols. Each task
    contributes n=len(group) samples and c=# successes within that group."""

    def _agg(g: pd.DataFrame) -> float:
        n = len(g)
        c = int((g["status"] == "success").sum())
        k_eff = min(k, n)
        return pass_at_k(n, c, k_eff)

    # pass@k is defined per-task (n samples of the *same* task); average across tasks.
    per_task = df.groupby(group_cols + ["task_id"]).apply(_agg, include_groups=False)
    return per_task.groupby(level=list(range(len(group_cols)))).mean()


def chi_square_model_vs_outcome(df: pd.DataFrame) -> tuple[float, float, int, pd.DataFrame]:
    """Chi-square test of independence between model and outcome status."""
    table = pd.crosstab(df["model"], df["status"])
    chi2, p, dof, _expected = stats.chi2_contingency(table)
    return chi2, p, dof, table


# --------------------------------------------------------------------------- tables


def summarize_by_model(df: pd.DataFrame, repeats: int) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        n = len(g)
        successes = int((g["status"] == "success").sum())
        lo, hi = wilson_ci(successes, n)
        pass1 = pass_at_k_grouped(g.assign(_dummy=1), ["_dummy"], k=1).iloc[0]
        passk = pass_at_k_grouped(g.assign(_dummy=1), ["_dummy"], k=repeats).iloc[0]
        successful = g[g["status"] == "success"]
        rows.append(
            {
                "model": model,
                "n_attempts": n,
                "pass_rate": successes / n,
                "wilson_lo": lo,
                "wilson_hi": hi,
                f"pass@1": pass1,
                f"pass@{repeats}": passk,
                "avg_prompt_tokens": g["prompt_tokens"].mean(),
                "avg_output_tokens": g["output_tokens"].mean(),
                "avg_latency_sec": g["total_duration_sec"].mean(),
                "avg_tokens_per_sec": g["tokens_per_sec"].mean(),
                "cost_per_success_usd": (g["cost_usd"].sum() / len(successful)) if len(successful) else float("nan"),
                "total_cost_usd": g["cost_usd"].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("pass_rate", ascending=False).reset_index(drop=True)


def summarize_by_model_task_type(df: pd.DataFrame, repeats: int) -> pd.DataFrame:
    rows = []
    for (model, task_type), g in df.groupby(["model", "task_type"]):
        n = len(g)
        successes = int((g["status"] == "success").sum())
        lo, hi = wilson_ci(successes, n)
        rows.append(
            {
                "model": model,
                "task_type": task_type,
                "n_attempts": n,
                "pass_rate": successes / n,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "avg_tokens_per_sec": g["tokens_per_sec"].mean(),
                "avg_cost_usd": g["cost_usd"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["task_type", "pass_rate"], ascending=[True, False]).reset_index(drop=True)


# ------------------------------------------------------------------------- markdown


def df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Hand-rolled markdown table renderer (no `tabulate` dependency available
    in this environment)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- chart


def make_pass_rate_chart(model_summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of pass rate by model with Wilson 95% CI error bars."""
    models = model_summary["model"].tolist()
    rates = model_summary["pass_rate"].to_numpy()
    lo = model_summary["wilson_lo"].to_numpy()
    hi = model_summary["wilson_hi"].to_numpy()
    err_lo = rates - lo
    err_hi = hi - rates

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = np.arange(len(models))
    accent = "#4C72B0"
    bars = ax.bar(x, rates, color=accent, width=0.55, zorder=3)
    ax.errorbar(x, rates, yerr=[err_lo, err_hi], fmt="none", ecolor="#333333", capsize=4, linewidth=1.2, zorder=4)

    for xi, r in zip(x, rates):
        ax.text(xi, r + 0.03, f"{r:.0%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Pass rate (status == success)")
    ax.set_ylim(0, 1.08)
    ax.set_title("LLM benchmark: pass rate by model (Wilson 95% CI)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(
            f"No results file at {args.results}. Run run_benchmark.py first "
            f"(add --mock to test without an Ollama server)."
        )

    df = pd.read_csv(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    repeats = int(df.groupby(["model", "task_id"]).size().max())

    model_summary = summarize_by_model(df, repeats)
    task_type_summary = summarize_by_model_task_type(df, repeats)
    chi2, p_value, dof, contingency = chi_square_model_vs_outcome(df)

    make_pass_rate_chart(model_summary, args.out_dir / "pass_rate_by_model.png")

    # --- rating vs status, if human-style comparison is meaningful here: we only
    # have objective test outcomes in this harness (no human_rating column), so
    # we instead report tests_passed/tests_total fraction as a proxy for partial
    # credit, which the main dataset's agent_runs table also tracks.
    df["test_fraction"] = df["tests_passed"] / df["tests_total"].replace(0, np.nan)
    avg_test_fraction_by_model = df.groupby("model")["test_fraction"].mean().sort_values(ascending=False)

    error_types = (
        df[df["status"] != "success"]["error_type"].fillna("(no code / no error)").value_counts().rename("count")
    )

    # --------------------------------------------------------------------- write it
    lines = []
    lines.append("# LLM Benchmark Summary\n")
    if df["model"].astype(str).str.startswith("MOCK/").any():
        lines.append(
            "> **SIMULATED DATA (--mock mode).** These rows come from the offline mock generator, "
            "not from real models. They only demonstrate the pipeline; run against a live Ollama "
            "server for real results.\n"
        )
    lines.append(
        f"Generated from `{args.results.relative_to(ROOT.parent) if ROOT.parent in args.results.parents else args.results}`. "
        f"{len(df)} attempts across {df['model'].nunique()} model(s), {df['task_id'].nunique()} task(s), "
        f"{repeats} repeat(s) per (model, task) pair.\n"
    )

    lines.append("## Pass@1 / Pass@k and cost by model\n")
    display_cols = [
        "model", "n_attempts", "pass_rate", "wilson_lo", "wilson_hi",
        "pass@1", f"pass@{repeats}", "avg_prompt_tokens", "avg_output_tokens",
        "avg_latency_sec", "avg_tokens_per_sec", "cost_per_success_usd", "total_cost_usd",
    ]
    lines.append(df_to_markdown(model_summary[display_cols]))
    lines.append(
        "\n`pass_rate` = fraction of attempts with status == success. `wilson_lo`/`wilson_hi` "
        "are the 95% Wilson score confidence interval on that rate (more reliable than a "
        "normal approximation at small sample sizes). `pass@1` and `pass@k` use the unbiased "
        "estimator from Chen et al. 2021: `1 - C(n-c, k) / C(n, k)`, averaged per task then "
        "across tasks. `cost_per_success_usd` = total simulated API-equivalent cost for the "
        "model divided by its count of fully-successful attempts.\n"
    )

    lines.append("## Pass rate by model x task_type (with Wilson 95% CI)\n")
    lines.append(
        df_to_markdown(
            task_type_summary[["model", "task_type", "n_attempts", "pass_rate", "wilson_lo", "wilson_hi", "avg_tokens_per_sec", "avg_cost_usd"]]
        )
    )
    lines.append("")

    lines.append("## Chi-square test: model vs. outcome (success/partial/failed)\n")
    lines.append(f"Contingency table (rows = model, columns = status):\n")
    lines.append(df_to_markdown(contingency.reset_index(), float_fmt="{:.0f}"))
    lines.append(
        f"\nChi2 = {chi2:.3f}, dof = {dof}, p-value = {p_value:.4g}. "
        + (
            "p < 0.05: outcome distribution differs significantly by model (models are not "
            "interchangeable on this task bank)."
            if p_value < 0.05
            else "p >= 0.05: no significant evidence that outcome distribution differs by model "
            "at this sample size — consider more repeats before concluding models are equivalent."
        )
        + "\n"
    )

    lines.append("## Average hidden-test pass fraction by model (partial-credit view)\n")
    lines.append(
        df_to_markdown(avg_test_fraction_by_model.reset_index().rename(columns={"test_fraction": "avg_test_pass_fraction"}))
    )
    lines.append("")

    lines.append("## Failure/error type breakdown (non-success attempts)\n")
    if len(error_types):
        lines.append(df_to_markdown(error_types.reset_index().rename(columns={"index": "error_type"}), float_fmt="{:.0f}"))
    else:
        lines.append("(no failures recorded)")
    lines.append("")

    lines.append("## Chart\n")
    lines.append("![Pass rate by model](pass_rate_by_model.png)\n")

    summary_path = args.out_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {args.out_dir / 'pass_rate_by_model.png'}")
    print("\n--- model summary (pass_rate) ---")
    print(model_summary[["model", "n_attempts", "pass_rate", "wilson_lo", "wilson_hi", "cost_per_success_usd"]].to_string(index=False))
    print(f"\nChi-square model vs outcome: chi2={chi2:.3f}, p={p_value:.4g}")


if __name__ == "__main__":
    main()
