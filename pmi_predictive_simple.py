#!/usr/bin/env python3
"""
Simplified PMI Predictive Pipeline (Plug-and-Play)
-------------------------------------------------
Edit paths below and run:
    python pmi_predictive_simple.py

Features:
- Reads membership.csv & claims.csv
- Creates member-level features & 2-year target
- Trains LightGBM regression
- Generates feature importance, SHAP, and risk segmentation

Author: ChatGPT
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import shap

# ----------- USER INPUTS -----------
MEMBERSHIP_PATH = "membership_synthetic_full.csv"
CLAIMS_PATH = "claims_synthetic_full.csv"
OUTPUT_DIR = "pmi_output"
CUTOFF_DATE = "2023-01-01"
RISK_METHOD = "threshold"  # or "percentile"
RISK_THRESHOLD = 10000.0
PCTILE1, PCTILE2 = 0.8, 0.95

# -----------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_cols(df):
    df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]
    return df

def ensure_date(df, col):
    return pd.to_datetime(df[col], errors="coerce")

def read_data():
    print("Reading data...")
    mem = sanitize_cols(pd.read_csv(MEMBERSHIP_PATH))
    claims = sanitize_cols(pd.read_csv(CLAIMS_PATH))
    return mem, claims

def feature_engineering(mem, claims):
    print("Building features...")
    cutoff = pd.to_datetime(CUTOFF_DATE)
    mem_key = [c for c in mem.columns if "unique" in c and "member" in c]
    claim_key = [c for c in claims.columns if "unique" in c and "member" in c]
    if not mem_key or not claim_key:
        raise ValueError("UniqueMemberRef not found in input files")
    mem = mem.rename(columns={mem_key[0]: "UniqueMemberRef"})
    claims = claims.rename(columns={claim_key[0]: "UniqueMemberRef"})
    
    date_col = [c for c in claims.columns if "date" in c]
    amt_col = [c for c in claims.columns if "amount" in c]
    claims["ClaimDate"] = ensure_date(claims, date_col[0])
    claims["ClaimAmount"] = pd.to_numeric(claims[amt_col[0]], errors="coerce")
    
    members = mem[["UniqueMemberRef"]].drop_duplicates()
    if "dateofbirth" in mem.columns:
        mem["DateOfBirth"] = pd.to_datetime(mem["dateofbirth"], errors="coerce")
        members = members.merge(mem[["UniqueMemberRef", "DateOfBirth"]], on="UniqueMemberRef", how="left")
        members["Age"] = (cutoff - members["DateOfBirth"]).dt.days // 365
    else:
        members["Age"] = np.random.randint(20, 70, len(members))
    
    # Claims summary (past 24 months)
    start = cutoff - pd.DateOffset(months=24)
    sub = claims[(claims["ClaimDate"] >= start) & (claims["ClaimDate"] < cutoff)]
    agg = sub.groupby("UniqueMemberRef").agg(
        num_claims=("ClaimAmount", "count"),
        total_amount=("ClaimAmount", "sum"),
        max_amount=("ClaimAmount", "max")
    ).reset_index()
    members = members.merge(agg, on="UniqueMemberRef", how="left").fillna(0)
    
    # Future target (next 24 months)
    future_end = cutoff + pd.DateOffset(months=24)
    fut = claims[(claims["ClaimDate"] >= cutoff) & (claims["ClaimDate"] < future_end)]
    tgt = fut.groupby("UniqueMemberRef")["ClaimAmount"].sum().reset_index().rename(columns={"ClaimAmount":"target"})
    members = members.merge(tgt, on="UniqueMemberRef", how="left").fillna(0)
    return members

def train_model(df):
    print("Training model...")
    X = df[["Age", "num_claims", "total_amount", "max_amount"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(objective="regression", learning_rate=0.05, n_estimators=300, num_leaves=64)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # compute RMSE robustly (works across sklearn versions)
    import numpy as _np
    rmse = float(_np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    print(f"RMSE={rmse:.2f} | R2={r2:.3f}")

    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi.plot(kind="barh", title="Feature Importance")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png")); plt.close()

    # SHAP (safe: sample and guard)
    try:
        print("Generating SHAP plots...")
        explainer = shap.TreeExplainer(model)
        sample_X = X_train.sample(min(500, len(X_train)), random_state=42)
        shap_values = explainer.shap_values(sample_X)
        shap.summary_plot(shap_values, sample_X, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png")); plt.close()
    except Exception as e:
        print("SHAP failed:", e)

    return model, X_test, y_test, preds


def risk_segmentation(df, preds):
    print("Assigning risk groups...")
    out = df.copy()
    out["predicted"] = preds
    if RISK_METHOD == "threshold":
        # dynamically handle thresholds
        max_pred = out["predicted"].max()
        if max_pred <= RISK_THRESHOLD:
            bins = [-1, RISK_THRESHOLD, max_pred + 1]
            labels = ["Low", "High"]
        elif max_pred <= 3 * RISK_THRESHOLD:
            bins = [-1, RISK_THRESHOLD, max_pred + 1]
            labels = ["Low", "High"]
        else:
            bins = [-1, RISK_THRESHOLD, 3 * RISK_THRESHOLD, max_pred + 1]
            labels = ["Low", "Medium", "High"]

        out["Risk"] = pd.cut(out["predicted"], bins=bins, labels=labels)

    else:
        q1 = out["predicted"].quantile(PCTILE1)
        q2 = out["predicted"].quantile(PCTILE2)
        out["Risk"] = pd.cut(out["predicted"],
                             bins=[-1, q1, q2, out["predicted"].max()+1],
                             labels=["Low","Medium","High"])
    out.to_csv(os.path.join(OUTPUT_DIR, "risk_segments.csv"), index=False)
    print("Risk segmentation saved!")

def main():
    mem, claims = read_data()
    df = feature_engineering(mem, claims)
    df.to_csv(os.path.join(OUTPUT_DIR, "features.csv"), index=False)
    model, X_test, y_test, preds = train_model(df)
    risk_segmentation(df, model.predict(df[["Age","num_claims","total_amount","max_amount"]]))
    print("Pipeline complete. All outputs saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
