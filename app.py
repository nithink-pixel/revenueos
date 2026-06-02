"""
RevenueOS - Executive Dashboard
---------------------------------
The main Streamlit application.
Run with: streamlit run app.py

Architecture:
  1. User uploads their data (or uses sample data)
  2. All three detection modules run in parallel
  3. Claude synthesizes findings into a decision brief
  4. Dashboard shows: top-line numbers → ranked leaks → AI brief → charts
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from modules import churn_radar, marketing_waste, support_bottleneck, ai_synthesis

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "RevenueOS",
    page_icon  = "💰",
    layout     = "wide",
)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px 20px;
    border: 1px solid #e9ecef;
}
.leak-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6c757d;
    margin-bottom: 4px;
}
.opportunity-badge {
    background: #d4edda;
    color: #155724;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("💰 RevenueOS")
st.markdown("**Automated revenue leak detection for SaaS businesses.**  "
            "Upload your data or use the sample dataset to find where money is leaving.")

st.divider()

# ── SIDEBAR: DATA SOURCE ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    use_sample = st.toggle("Use sample data", value=True)

    if use_sample:
        st.info("Using built-in SaaS sample data (500 customers, 12 months)")
        customers_path = "sample_data/customers.csv"
        marketing_path = "sample_data/marketing.csv"
        tickets_path   = "sample_data/support_tickets.csv"
    else:
        st.markdown("**Upload your CSV files:**")
        cust_file  = st.file_uploader("Customers CSV", type="csv")
        mkt_file   = st.file_uploader("Marketing Spend CSV", type="csv")
        tick_file  = st.file_uploader("Support Tickets CSV", type="csv")

        if cust_file:
            pd.read_csv(cust_file).to_csv("/tmp/customers.csv", index=False)
            customers_path = "/tmp/customers.csv"
        else:
            customers_path = "sample_data/customers.csv"

        if mkt_file:
            pd.read_csv(mkt_file).to_csv("/tmp/marketing.csv", index=False)
            marketing_path = "/tmp/marketing.csv"
        else:
            marketing_path = "sample_data/marketing.csv"

        if tick_file:
            pd.read_csv(tick_file).to_csv("/tmp/tickets.csv", index=False)
            tickets_path = "/tmp/tickets.csv"
        else:
            tickets_path = "sample_data/support_tickets.csv"

    st.divider()
    run_button = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
    generate_brief = st.button("🤖 Generate AI Brief", use_container_width=True)

# ── RUN ANALYSIS ─────────────────────────────────────────────────────────────
if run_button or "results" not in st.session_state:
    with st.spinner("Running leak detection engines..."):
        churn_r    = churn_radar.run(customers_path)
        mkt_r      = marketing_waste.run(marketing_path)
        support_r  = support_bottleneck.run(tickets_path, customers_path)

    st.session_state["results"] = {
        "churn":    churn_r,
        "mkt":      mkt_r,
        "support":  support_r,
    }

results = st.session_state.get("results")
if not results:
    st.stop()

churn_r   = results["churn"]
mkt_r     = results["mkt"]
support_r = results["support"]

# ── TOP LINE METRICS ─────────────────────────────────────────────────────────
total_opportunity = (
    churn_r["summary"]["total_arr_at_risk"] +
    mkt_r["summary"]["total_waste"] +
    support_r["summary"]["arr_at_risk"]
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label = "Total Revenue Opportunity",
        value = f"${total_opportunity:,.0f}",
        delta = "leaks identified",
    )
with c2:
    st.metric(
        label = "ARR at Churn Risk",
        value = f"${churn_r['summary']['total_arr_at_risk']:,.0f}",
        delta = f"{churn_r['summary']['findings_count']} segments flagged",
        delta_color = "inverse",
    )
with c3:
    st.metric(
        label = "Marketing Waste",
        value = f"${mkt_r['summary']['total_waste']:,.0f}",
        delta = f"CAC {mkt_r['summary']['worst_cac_ratio']}x blended avg",
        delta_color = "inverse",
    )
with c4:
    st.metric(
        label = "Support ARR at Risk",
        value = f"${support_r['summary']['arr_at_risk']:,.0f}",
        delta = f"{support_r['summary']['avg_response_hours']}h avg response",
        delta_color = "inverse",
    )

st.divider()

# ── RANKED LEAKS ─────────────────────────────────────────────────────────────
st.subheader("🚨 Revenue Leaks — Ranked by Impact")

# Combine all findings into one ranked list
all_findings = []

for f in churn_r["findings"]:
    all_findings.append({
        "rank":        0,
        "type":        "Churn Risk",
        "description": f"{f['group']} ({f['dimension']}): {f['churn_rate']}% churn rate ({f['ratio']}x baseline)",
        "impact":      f["impact_usd"],
        "action":      f"Investigate why {f['group']} customers churn at {f['churn_rate']}%. "
                      f"Start with exit interviews and support ticket review.",
    })

for f in mkt_r["findings"]:
    all_findings.append({
        "rank":        0,
        "type":        "Marketing Waste",
        "description": f"{f['channel']}: CAC ${f['cac']:,.0f} ({f['cac_ratio']}x blended average)",
        "impact":      f["impact_usd"],
        "action":      f"Reduce spend on {f['channel']} immediately. Reallocate budget to "
                      f"channels with CAC near ${f['blended_cac']:,.0f}.",
    })

for f in support_r["findings"]:
    all_findings.append({
        "rank":        0,
        "type":        "Support Bottleneck",
        "description": f"Slow support response (>{f['threshold_hours']}h) correlates with churn. "
                      f"Worst segment: {support_r['summary']['worst_segment']}",
        "impact":      f["impact_usd"],
        "action":      "Set SLA: all tickets resolved within 8h. "
                      "Route Healthcare and Enterprise tickets to senior agents immediately.",
    })

all_findings.sort(key=lambda x: x["impact"], reverse=True)

for i, finding in enumerate(all_findings):
    color_map = {
        "Churn Risk":         "🔴",
        "Marketing Waste":    "🟡",
        "Support Bottleneck": "🟠",
    }
    icon = color_map.get(finding["type"], "⚪")

    with st.expander(
        f"{icon} #{i+1} — {finding['type']} — ${finding['impact']:,.0f} at risk",
        expanded=(i == 0),
    ):
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"**Finding:** {finding['description']}")
            st.markdown(f"**Recommended action:** {finding['action']}")
        with col_b:
            st.metric("ARR Impact", f"${finding['impact']:,.0f}")

st.divider()

# ── CHARTS ───────────────────────────────────────────────────────────────────
st.subheader("📊 Detailed Analysis")

tab1, tab2, tab3 = st.tabs(["Churn Analysis", "Marketing Efficiency", "Support Performance"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        # Churn rate by segment
        df_customers = churn_r["raw"]
        seg_churn = (
            df_customers.groupby("segment")["churned"]
            .mean()
            .reset_index()
            .rename(columns={"churned": "churn_rate"})
            .sort_values("churn_rate", ascending=True)
        )
        seg_churn["churn_rate_pct"] = seg_churn["churn_rate"] * 100

        fig = px.bar(
            seg_churn,
            x        = "churn_rate_pct",
            y        = "segment",
            orientation = "h",
            title    = "Churn Rate by Segment",
            labels   = {"churn_rate_pct": "Churn Rate (%)", "segment": ""},
            color    = "churn_rate_pct",
            color_continuous_scale = "Reds",
        )
        fig.add_vline(
            x          = churn_r["summary"]["baseline_churn_rate"],
            line_dash  = "dash",
            line_color = "gray",
            annotation_text = f"Baseline {churn_r['summary']['baseline_churn_rate']}%",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn rate by plan
        plan_churn = (
            df_customers.groupby("plan")["churned"]
            .mean()
            .reset_index()
            .rename(columns={"churned": "churn_rate"})
        )
        plan_churn["churn_rate_pct"] = plan_churn["churn_rate"] * 100

        fig2 = px.bar(
            plan_churn,
            x      = "plan",
            y      = "churn_rate_pct",
            title  = "Churn Rate by Plan Type",
            labels = {"churn_rate_pct": "Churn Rate (%)", "plan": ""},
            color  = "churn_rate_pct",
            color_continuous_scale = "Oranges",
        )
        fig2.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        by_ch = mkt_r["by_channel"].copy()
        by_ch["blended"] = mkt_r["summary"]["blended_cac"]

        fig3 = px.bar(
            by_ch.sort_values("cac", ascending=True),
            x     = "cac",
            y     = "channel",
            orientation = "h",
            title = "Customer Acquisition Cost by Channel",
            labels = {"cac": "CAC ($)", "channel": ""},
            color  = "cac",
            color_continuous_scale = "YlOrRd",
        )
        fig3.add_vline(
            x          = mkt_r["summary"]["blended_cac"],
            line_dash  = "dash",
            line_color = "blue",
            annotation_text = f"Blended ${mkt_r['summary']['blended_cac']:,.0f}",
        )
        fig3.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.scatter(
            by_ch,
            x      = "total_spend",
            y      = "total_acquired",
            size   = "cac",
            color  = "cac",
            text   = "channel",
            title  = "Spend vs Customers Acquired (bubble = CAC)",
            labels = {
                "total_spend":    "Total Spend ($)",
                "total_acquired": "Customers Acquired",
                "cac":            "CAC",
            },
            color_continuous_scale = "Reds",
        )
        fig4.update_traces(textposition="top center")
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        bucket_df = support_r["bucket_churn"].copy()
        bucket_df["churn_rate_pct"] = bucket_df["churn_rate"] * 100

        fig5 = px.bar(
            bucket_df,
            x      = "response_bucket",
            y      = "churn_rate_pct",
            title  = "Churn Rate by Support Response Time",
            labels = {
                "response_bucket": "Response Time Bucket",
                "churn_rate_pct":  "Churn Rate (%)",
            },
            color  = "churn_rate_pct",
            color_continuous_scale = "Reds",
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        seg_sup = support_r["segment_support"]
        fig6 = px.scatter(
            seg_sup,
            x     = "avg_response_hours",
            y     = "churn_rate",
            size  = "tickets_per_cust",
            text  = "segment",
            title = "Support Response Time vs Churn Rate by Segment",
            labels = {
                "avg_response_hours": "Avg Response Time (hours)",
                "churn_rate":         "Churn Rate",
                "tickets_per_cust":   "Total Tickets",
            },
        )
        fig6.update_traces(textposition="top center")
        st.plotly_chart(fig6, use_container_width=True)
# ── ML CHURN PREDICTION ──────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 ML Churn Prediction (XGBoost + SHAP)")

if st.button("Run Churn Prediction Model", type="primary"):
    with st.spinner("Training XGBoost model and computing SHAP values..."):
        from modules import churn_prediction
        ml_result = churn_prediction.run(customers_path)
        st.session_state["ml"] = ml_result

if "ml" in st.session_state:
    ml = st.session_state["ml"]
    s  = ml["summary"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC Score",       s["auc_score"], "Model accuracy")
    col2.metric("F1 Score",        s["f1_score"])
    col3.metric("High Risk Customers", s["high_risk_count"])
    col4.metric("ARR at Risk",     f"${s['arr_at_risk']:,.0f}")

    tab_a, tab_b = st.tabs(["At-Risk Customers", "SHAP Feature Importance"])

    with tab_a:
        st.markdown("**Top 20 customers most likely to churn next — ranked by probability**")
        st.dataframe(
            ml["at_risk"].style.background_gradient(
                subset=["churn_probability"], cmap="Reds"
            ),
            use_container_width=True,
        )

    with tab_b:
        fig_shap = px.bar(
            ml["shap_importance"].head(10),
            x     = "importance",
            y     = "feature_label",
            orientation = "h",
            title = "Top Churn Drivers (SHAP Feature Importance)",
            labels = {"importance": "Mean |SHAP Value|", "feature_label": ""},
            color  = "importance",
            color_continuous_scale = "Reds",
        )
        fig_shap.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(f"Top driver: **{s['top_churn_driver']}** — {s['second_driver']} is second most important.")
st.divider()

# ── AI BRIEF ─────────────────────────────────────────────────────────────────
st.subheader("🤖 Executive Decision Brief")

if generate_brief or "brief" in st.session_state:
    if generate_brief or "brief" not in st.session_state:
        with st.spinner("Claude is analyzing your data and writing the brief..."):
            try:
                brief = ai_synthesis.run(churn_r, mkt_r, support_r)
                st.session_state["brief"] = brief
            except Exception as e:
                st.error(f"Could not generate brief: {e}. "
                         "Make sure your ANTHROPIC_API_KEY is set.")
                brief = None
    else:
        brief = st.session_state["brief"]

    if brief:
        st.markdown(brief)
else:
    st.info("Click **'Generate AI Brief'** in the sidebar to get an executive "
            "decision brief written by Claude.")
