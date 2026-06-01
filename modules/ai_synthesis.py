"""
RevenueOS - AI Synthesis Layer
--------------------------------
What this does:
  - Takes the raw output from all three detection modules
  - Calls the Claude API with a structured prompt
  - Gets back a plain-English executive decision brief
  - The brief includes: situation, root cause, dollar impact, ranked actions

Why this is the hardest part to replicate:
  Anybody can build a dashboard.
  The hard part is turning numbers into a decision.
  Claude acts as a McKinsey analyst here — it explains WHY the numbers
  matter, what's causing them, and what to do about it in order of ROI.
"""

import anthropic
import json


def build_prompt(churn_result: dict, marketing_result: dict, support_result: dict) -> str:
    """
    Constructs the structured data payload for Claude.
    We pass structured JSON so Claude can reason precisely about numbers.
    """
    payload = {
        "company_profile": {
            "total_arr":          churn_result["summary"]["total_arr"],
            "total_customers":    len(churn_result["raw"]),
            "baseline_churn":     churn_result["summary"]["baseline_churn_rate"],
            "avg_mrr_per_customer": churn_result["summary"]["avg_mrr"],
        },
        "leak_1_churn": {
            "total_arr_at_risk": churn_result["summary"]["total_arr_at_risk"],
            "anomalies": [
                {
                    "dimension":    f["dimension"],
                    "group":        f["group"],
                    "churn_rate":   f["churn_rate"],
                    "vs_baseline":  f["baseline_rate"],
                    "ratio":        f["ratio"],
                    "arr_at_risk":  f["arr_at_risk"],
                }
                for f in churn_result["findings"][:3]   # top 3 only
            ],
        },
        "leak_2_marketing": {
            "total_waste":   marketing_result["summary"]["total_waste"],
            "blended_cac":   marketing_result["summary"]["blended_cac"],
            "anomalies": [
                {
                    "channel":    f["channel"],
                    "cac":        f["cac"],
                    "vs_blended": f["blended_cac"],
                    "ratio":      f["cac_ratio"],
                    "waste":      f["wasted_spend"],
                }
                for f in marketing_result["findings"]
            ],
        },
        "leak_3_support": {
            "arr_at_risk":        support_result["summary"]["arr_at_risk"],
            "avg_response_hours": support_result["summary"]["avg_response_hours"],
            "slow_vs_fast_ratio": support_result["summary"]["slow_vs_fast_ratio"],
            "worst_segment":      support_result["summary"]["worst_segment"],
            "worst_segment_hours": support_result["summary"]["worst_segment_hours"],
        },
    }

    return f"""You are a senior business analyst reviewing data from a SaaS company's operations.
You have been given structured findings from three automated leak detection engines.
Your job is to write a concise executive decision brief that a CEO can act on today.

Here is the data:
{json.dumps(payload, indent=2)}

Write your response in this exact structure:

## Situation
One paragraph. What is the overall health of the business? Lead with the most important number.

## Top 3 Revenue Leaks (ranked by dollar impact)

### 1. [Name the leak — e.g. "Healthcare Segment Churn"]
- **Root cause diagnosis**: What is actually causing this? Be specific.
- **ARR impact**: $X at risk
- **Recommended action**: One concrete, specific action the team can take this week.

### 2. [Name the leak]
- **Root cause diagnosis**: ...
- **ARR impact**: $X at risk  
- **Recommended action**: ...

### 3. [Name the leak]
- **Root cause diagnosis**: ...
- **ARR impact**: $X at risk
- **Recommended action**: ...

## Total Revenue Opportunity
One sentence. Total ARR recoverable if all three issues are addressed.

## Priority This Week
Three bullet points. The three most important actions ranked by ease × impact.

Be direct. Use specific numbers. Do not hedge. Write like you are presenting to a board.
"""


def run(churn_result: dict, marketing_result: dict, support_result: dict) -> str:
    """
    Calls Claude and returns the executive brief as a string.
    """
    import streamlit as st

api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=api_key)

    prompt = build_prompt(churn_result, marketing_result, support_result)

    message = client.messages.create(
        model      = "claude-opus-4-5",
        max_tokens = 1024,
        messages   = [{"role": "user", "content": prompt}],
    )

    return message.content[0].text


if __name__ == "__main__":
    # Test the AI layer standalone using real module outputs
    import sys
    sys.path.insert(0, ".")
    from modules import churn_radar, marketing_waste, support_bottleneck

    print("Running detection engines...")
    churn_r    = churn_radar.run()
    marketing_r = marketing_waste.run()
    support_r  = support_bottleneck.run()

    print("Calling Claude for executive brief...\n")
    brief = run(churn_r, marketing_r, support_r)
    print(brief)
