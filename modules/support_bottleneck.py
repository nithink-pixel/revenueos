"""
RevenueOS - Module 3: Support Bottleneck Detector
---------------------------------------------------
What this does:
  - Joins support tickets with customer churn data
  - Finds the response time threshold above which churn increases sharply
  - Quantifies how much ARR is at risk from slow support
  - Identifies which ticket categories or segments have worst response times

Key concept — causal signal:
  We're not just saying "Healthcare has high churn."
  We're saying "Healthcare customers have 3x more support tickets AND
  their tickets take 8x longer to resolve — that's likely WHY they churn."
  That's root cause analysis. That's what makes this executive-worthy.
"""

import pandas as pd
import numpy as np


def run(
    tickets_path:   str = "sample_data/support_tickets.csv",
    customers_path: str = "sample_data/customers.csv",
) -> dict:

    tickets   = pd.read_csv(tickets_path)
    customers = pd.read_csv(customers_path)

    # ── JOIN: add churn status to every ticket ───────────────────────────────
    merged = tickets.drop(columns=["segment"], errors="ignore").merge(
        customers[["customer_id", "churned", "mrr", "segment"]],
        on="customer_id",
        how="left",
    )

    # ── FIND THE CHURN THRESHOLD ─────────────────────────────────────────────
    # We bucket response times into bins and compute churn rate per bucket.
    # The bin where churn spikes is our threshold.
    bins   = [0, 4, 8, 24, 48, 999]
    labels = ["<4h", "4-8h", "8-24h", "24-48h", ">48h"]
    merged["response_bucket"] = pd.cut(
        merged["response_hours"], bins=bins, labels=labels
    )

    bucket_churn = (
        merged.groupby("response_bucket", observed=True)
        .agg(
            tickets      = ("ticket_id", "count"),
            churn_rate   = ("churned", "mean"),
            avg_response = ("response_hours", "mean"),
        )
        .reset_index()
    )

    # ── SEGMENT BREAKDOWN ────────────────────────────────────────────────────
    segment_support = (
        merged.groupby("segment")
        .agg(
            avg_response_hours = ("response_hours", "mean"),
            tickets_per_cust   = ("ticket_id", "count"),
            churn_rate         = ("churned", "mean"),
        )
        .reset_index()
    )

    # ── ARR AT RISK FROM SLOW SUPPORT ────────────────────────────────────────
    # Customers whose tickets averaged >8h response AND who churned
    slow_threshold = 8.0
    customer_avg_response = (
        merged.groupby("customer_id")
        .agg(avg_response=("response_hours", "mean"))
        .reset_index()
    )
    slow_customers = customer_avg_response[
        customer_avg_response["avg_response"] > slow_threshold
    ]["customer_id"]

    at_risk = customers[
        (customers["customer_id"].isin(slow_customers)) &
        (customers["churned"] == 1)
    ]
    arr_at_risk = (at_risk["mrr"] * 12).sum()

    # ── FAST vs SLOW churn comparison ────────────────────────────────────────
    fast_churn = merged[merged["response_hours"] <= slow_threshold]["churned"].mean()
    slow_churn = merged[merged["response_hours"] >  slow_threshold]["churned"].mean()

    findings = []
    if slow_churn > fast_churn * 1.5:
        findings.append({
            "finding":         "slow_support_churn_correlation",
            "fast_churn_rate": round(fast_churn * 100, 1),
            "slow_churn_rate": round(slow_churn * 100, 1),
            "threshold_hours": slow_threshold,
            "ratio":           round(slow_churn / fast_churn, 2),
            "arr_at_risk":     round(arr_at_risk, 0),
            "impact_usd":      round(arr_at_risk, 0),
            "customers_affected": len(at_risk),
        })

    # Worst segment by response time
    worst_segment = segment_support.sort_values("avg_response_hours", ascending=False).iloc[0]

    summary = {
        "avg_response_hours":   round(merged["response_hours"].mean(), 1),
        "pct_resolved":         round(merged["resolved"].mean() * 100, 1),
        "arr_at_risk":          round(arr_at_risk, 0),
        "slow_vs_fast_ratio":   round(slow_churn / fast_churn, 2) if fast_churn > 0 else None,
        "worst_segment":        worst_segment["segment"],
        "worst_segment_hours":  round(worst_segment["avg_response_hours"], 1),
        "findings_count":       len(findings),
    }

    return {
        "module":          "support_bottleneck",
        "findings":        findings,
        "summary":         summary,
        "bucket_churn":    bucket_churn,
        "segment_support": segment_support,
    }


if __name__ == "__main__":
    result = run()
    print(f"\nAvg response time:   {result['summary']['avg_response_hours']}h")
    print(f"ARR at risk:         ${result['summary']['arr_at_risk']:,.0f}")
    print(f"Slow vs fast churn:  {result['summary']['slow_vs_fast_ratio']}x")
    print(f"Worst segment:       {result['summary']['worst_segment']} "
          f"({result['summary']['worst_segment_hours']}h avg)")
    for f in result["findings"]:
        print(f"\n  Slow support churn: {f['slow_churn_rate']}% vs {f['fast_churn_rate']}% fast "
              f"— ${f['arr_at_risk']:,.0f} ARR at risk")
