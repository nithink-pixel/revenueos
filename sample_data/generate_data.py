"""
RevenueOS - Sample Data Generator
Generates realistic SaaS business data with intentional leaks built in.
When you run this, it creates 3 CSV files that simulate a real SaaS company.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
N_CUSTOMERS   = 500
START_DATE    = datetime(2024, 1, 1)
END_DATE      = datetime(2024, 12, 31)


def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))


# ── 1. CUSTOMERS TABLE ───────────────────────────────────────────────────────
# This is the core table. Each row = one customer.
# We intentionally build in problems:
#   - Healthcare segment has very high churn (the leak)
#   - Monthly plan customers churn more than annual
#   - Small customers have lower LTV but huge volume

segments   = ["Enterprise", "Mid-Market", "SMB", "Healthcare", "Startup"]
plans      = ["Annual", "Monthly", "Quarterly"]
channels   = ["Organic", "Paid Search", "Referral", "Sales Outbound", "Partnership"]
regions    = ["North America", "Europe", "APAC", "LATAM"]

rows = []
for i in range(1, N_CUSTOMERS + 1):
    segment = random.choices(
        segments, weights=[15, 25, 30, 15, 15]
    )[0]
    plan = random.choices(plans, weights=[45, 35, 20])[0]
    channel = random.choices(
        channels, weights=[30, 25, 20, 15, 10]
    )[0]
    region = random.choices(
        regions, weights=[50, 25, 15, 10]
    )[0]

    # MRR varies by segment
    mrr_map = {
        "Enterprise":  (2000, 8000),
        "Mid-Market":  (500,  2000),
        "SMB":         (50,   500),
        "Healthcare":  (800,  4000),
        "Startup":     (50,   300),
    }
    lo, hi = mrr_map[segment]
    mrr = round(random.uniform(lo, hi), 2)

    signup_date = random_date(START_DATE, END_DATE - timedelta(days=90))

    # Churn probability — Healthcare and Monthly plans churn more (the LEAK)
    churn_prob = 0.10
    if segment == "Healthcare":
        churn_prob = 0.35   # <-- intentional leak: 3.5x normal churn
    if plan == "Monthly":
        churn_prob += 0.15
    if channel == "Paid Search":
        churn_prob += 0.08  # paid search customers are lower quality
    if region == "LATAM":
        churn_prob += 0.05

    churned = random.random() < churn_prob
    churn_date = None
    if churned:
        days_active = random.randint(30, 180)
        churn_date = signup_date + timedelta(days=days_active)
        if churn_date > END_DATE:
            churn_date = END_DATE
            churned = False

    support_tickets = random.randint(0, 3)
    if segment == "Healthcare":
        support_tickets += random.randint(2, 5)  # healthcare has more tickets too

    rows.append({
        "customer_id":     f"CUST-{i:04d}",
        "segment":         segment,
        "plan":            plan,
        "channel":         channel,
        "region":          region,
        "mrr":             mrr,
        "signup_date":     signup_date.strftime("%Y-%m-%d"),
        "churned":         int(churned),
        "churn_date":      churn_date.strftime("%Y-%m-%d") if churn_date else None,
        "support_tickets": support_tickets,
        "nps_score":       random.randint(1, 10),
        "login_frequency": random.randint(1, 30),  # logins per month
    })

customers = pd.DataFrame(rows)
customers.to_csv("sample_data/customers.csv", index=False)
print(f"customers.csv — {len(customers)} rows")


# ── 2. MARKETING SPEND TABLE ─────────────────────────────────────────────────
# Monthly marketing spend by channel.
# Intentional leak: Paid Social has terrible CAC (4x higher than Organic).

marketing_rows = []
months = pd.date_range("2024-01", "2024-12", freq="MS")
channels_mkt = ["Organic SEO", "Paid Search", "Paid Social", "Referral Program", "Events"]

for month in months:
    for ch in channels_mkt:
        spend_map = {
            "Organic SEO":      (5000,  8000),
            "Paid Search":      (15000, 30000),
            "Paid Social":      (20000, 40000),  # <-- high spend
            "Referral Program": (2000,  5000),
            "Events":           (8000,  15000),
        }
        lo, hi = spend_map[ch]
        spend = round(random.uniform(lo, hi), 2)

        # New customers acquired per channel per month
        # Paid Social gets lots of spend but few conversions = high CAC
        acq_map = {
            "Organic SEO":      (8,  15),
            "Paid Search":      (10, 20),
            "Paid Social":      (2,  6),   # <-- terrible conversion = leak
            "Referral Program": (5,  12),
            "Events":           (4,  10),
        }
        lo2, hi2 = acq_map[ch]
        new_customers = random.randint(lo2, hi2)

        marketing_rows.append({
            "month":         month.strftime("%Y-%m"),
            "channel":       ch,
            "spend":         spend,
            "new_customers": new_customers,
            "cac":           round(spend / max(new_customers, 1), 2),
        })

marketing = pd.DataFrame(marketing_rows)
marketing.to_csv("sample_data/marketing.csv", index=False)
print(f"marketing.csv   — {len(marketing)} rows")


# ── 3. SUPPORT TICKETS TABLE ─────────────────────────────────────────────────
# Individual support tickets.
# Intentional leak: slow response times correlate with churn.

ticket_rows = []
priorities   = ["Low", "Medium", "High", "Critical"]
categories   = ["Billing", "Technical", "Onboarding", "Feature Request", "Bug"]

churned_customers = customers[customers["churned"] == 1]["customer_id"].tolist()

for _, cust in customers.iterrows():
    n_tickets = cust["support_tickets"]
    for _ in range(n_tickets):
        ticket_date = random_date(
            datetime.strptime(cust["signup_date"], "%Y-%m-%d"),
            END_DATE
        )
        priority = random.choices(
            priorities, weights=[40, 35, 15, 10]
        )[0]

        # Churned customers had slower response times (or: slow response CAUSED churn)
        if cust["customer_id"] in churned_customers:
            response_hours = random.uniform(12, 72)  # slow = leak
        else:
            response_hours = random.uniform(1, 8)    # fast = healthy

        resolved = response_hours < 48
        ticket_rows.append({
            "ticket_id":      f"TKT-{len(ticket_rows)+1:05d}",
            "customer_id":    cust["customer_id"],
            "segment":        cust["segment"],
            "date":           ticket_date.strftime("%Y-%m-%d"),
            "priority":       priority,
            "category":       random.choice(categories),
            "response_hours": round(response_hours, 1),
            "resolved":       int(resolved),
        })

tickets = pd.DataFrame(ticket_rows)
tickets.to_csv("sample_data/support_tickets.csv", index=False)
print(f"support_tickets.csv — {len(tickets)} rows")

print("\nAll sample data generated. Files are in sample_data/")
print("\nIntentional leaks built into the data:")
print("  1. Healthcare segment: 35% churn rate vs 10% baseline")
print("  2. Paid Social: 4-8x higher CAC than other channels")
print("  3. Slow support response (12-72h) correlates with churn")
