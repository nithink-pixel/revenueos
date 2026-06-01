# RevenueOS

Automated revenue leak detection for SaaS businesses. Upload your data and find exactly where money is leaving — ranked by dollar impact — with an AI-generated executive action brief.

## What it detects

| Module | What it finds | Output |
|--------|--------------|--------|
| Churn Radar | Customer segments with abnormally high churn | ARR at risk per segment |
| Marketing Waste Detector | Channels with CAC above blended average | Wasted spend per channel |
| Support Bottleneck | Support response time → churn correlation | ARR at risk from slow support |

## Setup

```bash
# 1. Clone or download the project
cd revenueos

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# 4. Generate sample data
python sample_data/generate_data.py

# 5. Run the app
streamlit run app.py
```

## File structure

```
revenueos/
├── app.py                          # Main Streamlit dashboard
├── modules/
│   ├── churn_radar.py              # Churn anomaly detection
│   ├── marketing_waste.py          # CAC efficiency analysis
│   ├── support_bottleneck.py       # Support → churn correlation
│   └── ai_synthesis.py             # Claude executive brief generator
├── sample_data/
│   ├── generate_data.py            # Generates realistic test data
│   ├── customers.csv               # 500 customers with churn labels
│   ├── marketing.csv               # Monthly spend by channel
│   └── support_tickets.csv         # Individual ticket records
└── README.md
```

## Data format

If you bring your own data, the CSVs need these columns:

**customers.csv**: `customer_id, segment, plan, channel, region, mrr, signup_date, churned, churn_date, support_tickets, nps_score, login_frequency`

**marketing.csv**: `month, channel, spend, new_customers, cac`

**support_tickets.csv**: `ticket_id, customer_id, segment, date, priority, category, response_hours, resolved`

## Tech stack

- **Python** — data ingestion, cleaning, anomaly detection
- **Pandas / NumPy** — cohort analysis, CAC computation, correlation
- **Plotly** — interactive charts
- **Streamlit** — executive dashboard UI
- **Anthropic Claude API** — executive decision brief generation
