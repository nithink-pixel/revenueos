"""
RevenueOS - Module 1: Churn Radar
----------------------------------
What this does:
  - Loads the customers CSV
  - Calculates churn rate by every dimension (segment, plan, channel, region)
  - Finds which dimension has ABNORMALLY high churn vs the baseline
  - Quantifies the ARR at risk in dollars
  - Returns a structured result dict that the main app and Claude can consume

Why this matters for the portfolio:
  - Shows cohort analysis (not just raw ML)
  - Shows anomaly detection at business level
  - Quantifies financial impact — not just "churn is high", but "$840K at risk"
"""

import pandas as pd
import numpy as np


def run(customers_path: str = "sample_data/customers.csv") -> dict:
    """
    Main entry point. Returns a dict with:
      - findings: list of anomalies found, each with name, metric, impact_usd
      - summary:  top-line numbers for the dashboard
      - raw:      the full dataframe for deeper analysis
    """
    df = pd.read_csv(customers_path)

    # ── BASELINE ────────────────────────────────────────────────────────────
    # Overall churn rate across all customers. This is our benchmark.
    # Any segment more than 1.5x this is flagged as a leak.
    baseline_churn = df["churned"].mean()
    avg_mrr        = df["mrr"].mean()
    total_arr      = df["mrr"].sum() * 12  # Annual Run Rate

    findings = []

    # ── SCAN EVERY DIMENSION ────────────────────────────────────────────────
    # We loop through each categorical column and compute churn rate per group.
    # This is the core of cohort analysis — you're asking:
    # "Is churn uniformly distributed, or is it concentrated somewhere?"

    dimensions = ["segment", "plan", "channel", "region"]

    for dim in dimensions:
        group = (
            df.groupby(dim)
            .agg(
                customers   = ("customer_id", "count"),
                churned     = ("churned", "sum"),
                churn_rate  = ("churned", "mean"),
                avg_mrr     = ("mrr", "mean"),
                total_mrr   = ("mrr", "sum"),
            )
            .reset_index()
        )
        group["arr"] = group["total_mrr"] * 12

        for _, row in group.iterrows():
            if row["customers"] < 10:           # ignore tiny groups (noise)
                continue
            ratio = row["churn_rate"] / baseline_churn
            if ratio >= 1.5:                    # 50% worse than baseline = a leak
                # How much ARR is at risk?
                # = churned customers in this group * avg ARR per customer
                churned_count = row["churned"]
                arr_at_risk   = churned_count * row["avg_mrr"] * 12

                findings.append({
                    "dimension":      dim,
                    "group":          str(row[dim]),
                    "churn_rate":     round(row["churn_rate"] * 100, 1),
                    "baseline_rate":  round(baseline_churn * 100, 1),
                    "ratio":          round(ratio, 2),
                    "customers":      int(row["customers"]),
                    "churned_count":  int(row["churned"]),
                    "arr_at_risk":    round(arr_at_risk, 0),
                    "avg_mrr":        round(row["avg_mrr"], 0),
                    "impact_usd":     round(arr_at_risk, 0),
                })

    # Sort by dollar impact — most expensive leak first
    findings.sort(key=lambda x: x["impact_usd"], reverse=True)

    # ── SUPPORT CORRELATION ─────────────────────────────────────────────────
    # Extra signal: do customers with more support tickets churn more?
    ticket_corr = df[["support_tickets", "churned"]].corr().iloc[0, 1]

    summary = {
        "baseline_churn_rate": round(baseline_churn * 100, 1),
        "total_arr":           round(total_arr, 0),
        "total_arr_at_risk":   round(sum(f["arr_at_risk"] for f in findings), 0),
        "findings_count":      len(findings),
        "support_churn_corr":  round(ticket_corr, 3),
        "avg_mrr":             round(avg_mrr, 0),
    }

    return {
        "module":   "churn_radar",
        "findings": findings,
        "summary":  summary,
        "raw":      df,
    }


if __name__ == "__main__":
    result = run()
    print(f"\nBaseline churn rate: {result['summary']['baseline_churn_rate']}%")
    print(f"Total ARR:           ${result['summary']['total_arr']:,.0f}")
    print(f"ARR at risk:         ${result['summary']['total_arr_at_risk']:,.0f}")
    print(f"\nLeaks found ({result['summary']['findings_count']}):")
    for f in result["findings"]:
        print(f"  [{f['dimension']}] {f['group']}: {f['churn_rate']}% churn "
              f"({f['ratio']}x baseline) — ${f['arr_at_risk']:,.0f} ARR at risk")
