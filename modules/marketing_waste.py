"""
RevenueOS - Module 2: Marketing Waste Detector
------------------------------------------------
What this does:
  - Loads the marketing spend CSV
  - Calculates CAC (Customer Acquisition Cost) per channel
  - Finds channels where CAC is significantly above the blended average
  - Estimates how much money is being wasted vs if that spend had average efficiency
  - Returns structured findings

Key concept — CAC efficiency:
  CAC = Spend / New Customers Acquired
  If one channel's CAC is 4x the blended average,
  you're paying 4x more per customer than you need to.
  The waste = extra spend above what you'd pay at average efficiency.
"""

import pandas as pd
import numpy as np


def run(marketing_path: str = "sample_data/marketing.csv") -> dict:
    df = pd.read_csv(marketing_path)

    # ── BLENDED CAC ─────────────────────────────────────────────────────────
    # Blended CAC = total spend / total new customers across all channels.
    # This is the benchmark every channel is measured against.
    total_spend    = df["spend"].sum()
    total_acquired = df["new_customers"].sum()
    blended_cac    = total_spend / total_acquired

    # ── CAC BY CHANNEL ──────────────────────────────────────────────────────
    by_channel = (
        df.groupby("channel")
        .agg(
            total_spend    = ("spend", "sum"),
            total_acquired = ("new_customers", "sum"),
        )
        .reset_index()
    )
    by_channel["cac"]         = by_channel["total_spend"] / by_channel["total_acquired"]
    by_channel["cac_ratio"]   = by_channel["cac"] / blended_cac

    # ── MONTHLY TREND ───────────────────────────────────────────────────────
    monthly = (
        df.groupby(["month", "channel"])
        .agg(spend=("spend", "sum"), new_customers=("new_customers", "sum"))
        .reset_index()
    )
    monthly["cac"] = monthly["spend"] / monthly["new_customers"]

    findings = []

    for _, row in by_channel.iterrows():
        if row["cac_ratio"] >= 1.5:   # 50% worse than blended = waste
            # Waste = what you spent ABOVE what you'd have spent at blended CAC
            customers_acquired = row["total_acquired"]
            fair_spend         = customers_acquired * blended_cac
            wasted_spend       = row["total_spend"] - fair_spend

            # Annualize: our data is 12 months so this is already annual
            findings.append({
                "channel":          row["channel"],
                "cac":              round(row["cac"], 0),
                "blended_cac":      round(blended_cac, 0),
                "cac_ratio":        round(row["cac_ratio"], 2),
                "total_spend":      round(row["total_spend"], 0),
                "customers_acquired": int(row["total_acquired"]),
                "wasted_spend":     round(wasted_spend, 0),
                "impact_usd":       round(wasted_spend, 0),
            })

    findings.sort(key=lambda x: x["impact_usd"], reverse=True)

    summary = {
        "blended_cac":       round(blended_cac, 0),
        "total_spend":       round(total_spend, 0),
        "total_acquired":    int(total_acquired),
        "total_waste":       round(sum(f["wasted_spend"] for f in findings), 0),
        "findings_count":    len(findings),
        "worst_channel":     findings[0]["channel"] if findings else None,
        "worst_cac_ratio":   findings[0]["cac_ratio"] if findings else None,
    }

    return {
        "module":      "marketing_waste",
        "findings":    findings,
        "summary":     summary,
        "by_channel":  by_channel,
        "monthly":     monthly,
    }


if __name__ == "__main__":
    result = run()
    print(f"\nBlended CAC:   ${result['summary']['blended_cac']:,.0f}")
    print(f"Total spend:   ${result['summary']['total_spend']:,.0f}")
    print(f"Total waste:   ${result['summary']['total_waste']:,.0f}")
    print(f"\nWasteful channels ({result['summary']['findings_count']}):")
    for f in result["findings"]:
        print(f"  {f['channel']}: CAC ${f['cac']:,.0f} ({f['cac_ratio']}x blended) "
              f"— ${f['wasted_spend']:,.0f} wasted")
