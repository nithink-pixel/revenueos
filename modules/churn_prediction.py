"""
RevenueOS - Module 4: Churn Prediction Engine (XGBoost + SHAP)
---------------------------------------------------------------
What this does:
  - Engineers features from raw customer data
  - Trains XGBoost to predict individual churn probability
  - Uses SHAP to explain which features drive churn most
  - Returns ranked at-risk customers with probability scores
  - Returns model performance metrics (AUC, F1)

Portfolio value:
  - AUC score = the number data science JDs ask for
  - SHAP = explainable AI = what business stakeholders need
  - Forward-looking: predicts WHO will churn, not just WHO did
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")


def run(customers_path: str = "sample_data/customers.csv") -> dict:
    df = pd.read_csv(customers_path)

    # ── FEATURE ENGINEERING ─────────────────────────────────────────────────
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["days_active"]    = (pd.Timestamp("2024-12-31") - df["signup_date"]).dt.days
    df["ticket_rate"]    = df["support_tickets"] / (df["days_active"] / 30 + 1)
    df["low_nps"]        = (df["nps_score"] <= 4).astype(int)
    df["low_logins"]     = (df["login_frequency"] <= 5).astype(int)
    df["is_monthly"]     = (df["plan"] == "Monthly").astype(int)
    df["is_healthcare"]  = (df["segment"] == "Healthcare").astype(int)
    df["mrr_log"]        = np.log1p(df["mrr"])

    for col in ["segment", "plan", "channel", "region"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])

    feature_cols = [
        "mrr_log", "support_tickets", "nps_score", "login_frequency",
        "days_active", "ticket_rate", "low_nps", "low_logins",
        "is_monthly", "is_healthcare",
        "segment_enc", "plan_enc", "channel_enc", "region_enc",
    ]

    feature_labels = {
        "mrr_log":          "Monthly Revenue",
        "support_tickets":  "Support Tickets",
        "nps_score":        "NPS Score",
        "login_frequency":  "Login Frequency",
        "days_active":      "Days as Customer",
        "ticket_rate":      "Ticket Rate (per month)",
        "low_nps":          "Low NPS Flag",
        "low_logins":       "Low Login Flag",
        "is_monthly":       "Monthly Plan",
        "is_healthcare":    "Healthcare Segment",
        "segment_enc":      "Segment",
        "plan_enc":         "Plan Type",
        "channel_enc":      "Acquisition Channel",
        "region_enc":       "Region",
    }

    X = df[feature_cols]
    y = df["churned"]

    # ── TRAIN / TEST SPLIT ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── TRAIN XGBOOST ────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.02,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        scale_pos_weight  = (y == 0).sum() / (y == 1).sum(),
        random_state      = 42,
        eval_metric       = "auc",
        verbosity         = 0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ── EVALUATE ─────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)
    f1     = f1_score(y_test, y_pred)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_importance = pd.DataFrame({
        "feature":       feature_cols,
        "feature_label": [feature_labels[f] for f in feature_cols],
        "importance":    np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── SCORE ALL CUSTOMERS ──────────────────────────────────────────────────
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["risk_tier"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    at_risk = (
        df[df["churned"] == 0]
        .sort_values("churn_probability", ascending=False)
        .head(20)
        [["customer_id", "segment", "plan", "mrr",
          "churn_probability", "risk_tier", "support_tickets",
          "nps_score", "login_frequency"]]
        .copy()
    )
    at_risk["churn_probability"] = at_risk["churn_probability"].round(3)
    at_risk["arr"] = (at_risk["mrr"] * 12).round(0)

    high_risk   = df[(df["churned"] == 0) & (df["churn_probability"] >= 0.6)]
    arr_at_risk = (high_risk["mrr"] * 12).sum()

    summary = {
        "auc_score":          round(auc, 3),
        "f1_score":           round(f1, 3),
        "arr_at_risk":        round(arr_at_risk, 0),
        "high_risk_count":    len(high_risk),
        "medium_risk_count":  len(df[
            (df["churned"] == 0) &
            (df["churn_probability"] >= 0.3) &
            (df["churn_probability"] < 0.6)
        ]),
        "top_churn_driver":   shap_importance.iloc[0]["feature_label"],
        "second_driver":      shap_importance.iloc[1]["feature_label"],
    }

    return {
        "module":          "churn_prediction",
        "model":           model,
        "summary":         summary,
        "shap_importance": shap_importance,
        "at_risk":         at_risk,
        "shap_values":     shap_values,
        "feature_cols":    feature_cols,
        "feature_labels":  feature_labels,
        "X":               X,
        "df":              df,
    }


if __name__ == "__main__":
    result = run()
    s = result["summary"]
    print(f"\n{'='*40}")
    print(f"  Churn Prediction Model Results")
    print(f"{'='*40}")
    print(f"  AUC Score:        {s['auc_score']}  (higher = better, 1.0 = perfect)")
    print(f"  F1 Score:         {s['f1_score']}")
    print(f"  High risk:        {s['high_risk_count']} customers")
    print(f"  ARR at risk:      ${s['arr_at_risk']:,.0f}")
    print(f"  Top churn driver: {s['top_churn_driver']}")
    print(f"\nTop 5 at-risk customers:")
    print(result["at_risk"][["customer_id","segment","churn_probability","arr"]].head().to_string())
    print(f"\nTop 5 churn drivers (SHAP):")
    print(result["shap_importance"][["feature_label","importance"]].head().to_string())
