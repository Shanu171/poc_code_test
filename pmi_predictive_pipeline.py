#!/usr/bin/env python3
"""
PMI Predictive Pipeline (Plug-and-Play)

Save as pmi_predictive_pipeline.py and run:
python pmi_predictive_pipeline.py --membership /path/membership.csv --claims /path/claims.csv --output_dir ./pmi_out

Features:
- Loads membership and claims CSVs
- Builds member-level features and target: total claim cost for next N months (configurable)
- Preprocesses data, trains LightGBM with Optuna tuning (optional)
- Produces SHAP explanations, feature importance, diagnostic plots
- Allows user to set risk definition via amount threshold OR percentile cutoffs for High/Medium/Low

Dependencies:
  pandas, numpy, scikit-learn, lightgbm, xgboost, shap, optuna, matplotlib, joblib, category_encoders (optional)

Author: ChatGPT (robust template)
"""

import os, sys, argparse, logging, joblib, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import optuna
import shap

warnings.filterwarnings("ignore")

# ------------------ Utilities ------------------
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(output_dir, "pipeline.log"))]
    )
    return logging.getLogger("pmi_pipeline")

def safe_read_csv(path):
    return pd.read_csv(path, low_memory=False)

def ensure_date(df, col):
    return pd.to_datetime(df[col], errors="coerce")

def sanitize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").replace("/", "_").replace("-", "_") for c in df.columns]
    return df

def save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

# ------------------ Feature engineering ------------------
def prepare_member_level(membership_df, claims_df, cutoff_date_str, horizon_months=24, windows_months=(6,12,24), logger=None):
    cutoff = pd.to_datetime(cutoff_date_str)
    membership = sanitize_cols(membership_df)
    claims = sanitize_cols(claims_df)

    # detect keys
    mem_key = None
    for c in membership.columns:
        if "unique" in c.lower() and "member" in c.lower():
            mem_key = c; break
    claim_key = None
    for c in claims.columns:
        if "unique" in c.lower() and "member" in c.lower():
            claim_key = c; break
    if mem_key is None or claim_key is None:
        raise ValueError("Unique member ref not found in one of the files. Rename that column to include 'unique' and 'member'.")

    membership = membership.rename(columns={mem_key: "UniqueMemberRef"})
    claims = claims.rename(columns={claim_key: "UniqueMemberRef"})

    # standardize dates and amounts
    date_cols_claim = [c for c in claims.columns if "claim" in c.lower() and "date" in c.lower()]
    if not date_cols_claim:
        date_cols_claim = [c for c in claims.columns if "date" in c.lower()]
    claims["ClaimDate"] = ensure_date(claims, date_cols_claim[0])
    amount_cols = [c for c in claims.columns if "claim" in c.lower() and "amount" in c.lower()]
    if not amount_cols:
        amount_cols = [c for c in claims.columns if "amount" in c.lower()]
    claims["ClaimAmount"] = pd.to_numeric(claims[amount_cols[0]], errors="coerce")

    # prepare base member table
    members = membership[["UniqueMemberRef"]].drop_duplicates().reset_index(drop=True)
    # bring demographic columns if present
    for col in ["gender","plan","product","clientidentifier","postcode","dateofbirth","joindate"]:
        if col in membership.columns:
            members = members.merge(membership[["UniqueMemberRef", col]].drop_duplicates("UniqueMemberRef"), on="UniqueMemberRef", how="left")

    # compute age & tenure
    if "dateofbirth" in members.columns:
        members["DateOfBirth"] = pd.to_datetime(members["dateofbirth"], errors="coerce")
        members["AgeAtCutoff"] = (pd.to_datetime(cutoff) - members["DateOfBirth"]).dt.days // 365
    else:
        members["AgeAtCutoff"] = np.nan
    if "joindate" in members.columns:
        members["JoinDate"] = pd.to_datetime(members["joindate"], errors="coerce")
        members["TenureYears"] = (pd.to_datetime(cutoff) - members["JoinDate"]).dt.days // 365
    else:
        members["TenureYears"] = np.nan

    # historical window aggregations
    for w in windows_months:
        start = pd.to_datetime(cutoff) - pd.DateOffset(months=w)
        sub = claims[(claims["ClaimDate"] >= start) & (claims["ClaimDate"] < pd.to_datetime(cutoff))]
        agg = sub.groupby("UniqueMemberRef").agg(
            **{f"hist_{w}m_num_claims": ("ClaimAmount", "count"),
               f"hist_{w}m_total_amount": ("ClaimAmount", "sum"),
               f"hist_{w}m_max_amount": ("ClaimAmount", "max")}
        ).reset_index()
        members = members.merge(agg, on="UniqueMemberRef", how="left")
        for c in [f"hist_{w}m_num_claims", f"hist_{w}m_total_amount", f"hist_{w}m_max_amount"]:
            if c in members.columns:
                members[c] = members[c].fillna(0)

    # diagnosis flags example (if diagnosis column exists)
    diag_cols = [c for c in claims.columns if "diagnos" in c.lower() or "icd" in c.lower()]
    if diag_cols:
        dcol = diag_cols[0]
        claims["_diag_pre"] = claims[dcol].astype(str).str[:3]
        chronic_map = {"diabetes": ["E11","E10"], "hypertension": ["I10","I11"], "respiratory":["J45","J18"]}
        for k, prefixes in chronic_map.items():
            tmp = claims[claims["_diag_pre"].isin(prefixes)].groupby("UniqueMemberRef").size().reset_index(name=f"diag_{k}_count")
            members = members.merge(tmp, on="UniqueMemberRef", how="left")
            members[f"diag_{k}_count"] = members[f"diag_{k}_count"].fillna(0)
        claims = claims.drop(columns=["_diag_pre"], errors="ignore")

    # derived features
    if "hist_24m_num_claims" in members.columns:
        members["claims_per_year_24m"] = members["hist_24m_num_claims"] / 2.0
    # future target
    future_start = pd.to_datetime(cutoff)
    future_end = future_start + pd.DateOffset(months=horizon_months)
    future = claims[(claims["ClaimDate"] >= future_start) & (claims["ClaimDate"] < future_end)]
    tgt = future.groupby("UniqueMemberRef")["ClaimAmount"].sum().reset_index().rename(columns={"ClaimAmount":"target_future_cost"})
    members = members.merge(tgt, on="UniqueMemberRef", how="left")
    members["target_future_cost"] = members["target_future_cost"].fillna(0.0)
    return members, claims

# ------------------ Preprocess and split ------------------
def preprocess_and_split(members_df, target_col="target_future_cost", test_size=0.2, random_state=42):
    df = members_df.copy()
    ids = df["UniqueMemberRef"].copy()
    if "MemberName" in df.columns:
        df = df.drop(columns=["MemberName"])
    for c in [c for c in df.columns if "date" in c.lower() and "of" in c.lower() is False]:
        if c in df.columns and c not in ["DateOfBirth","JoinDate"]:
            try: df = df.drop(columns=[c])
            except: pass
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cat_cols = [c for c in df.columns if c not in num_cols and c != target_col and c != "UniqueMemberRef"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    enc_map = {}
    for c in cat_cols:
        df[c] = df[c].fillna("MISSING").astype(str)
        df[c], enc_map[c] = pd.factorize(df[c])
    X = df.drop(columns=[target_col, "UniqueMemberRef"])
    y = df[target_col].values
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(X, y, ids, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, ids_train, ids_test, enc_map

# ------------------ Modeling ------------------
def objective_lgb(trial, X_tr, y_tr, X_val, y_val):
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 512),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10)
    }
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    model = lgb.train(param, dtrain, valid_sets=[dval], num_boost_round=1000, early_stopping_rounds=50, verbose_eval=False)
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    return mean_squared_error(y_val, preds, squared=False)

def tune_lgb(X_train, y_train, X_val, y_val, n_trials=20):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    func = lambda trial: objective_lgb(trial, X_train, y_train, X_val, y_val)
    study.optimize(func, n_trials=n_trials)
    return study.best_trial.params

def train_lgb_final(X, y, params=None):
    if params is None:
        params = {"objective":"regression","metric":"rmse","verbosity":-1,"boosting_type":"gbdt","seed":42,"learning_rate":0.05,"num_leaves":128,"min_data_in_leaf":50,"feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5}
    dtrain = lgb.Dataset(X, y)
    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtrain], early_stopping_rounds=100)
    return model

# ------------------ SHAP and plots ------------------
def plot_feature_importance(model, feature_names, out_path):
    try:
        imp = model.feature_importance(importance_type="gain")
    except Exception:
        imp = model.feature_importance()
    idx = np.argsort(imp)[::-1]
    names = [feature_names[i] for i in idx]
    vals = imp[idx]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names)*0.15)))
    ax.barh(names[:30][::-1], vals[:30][::-1])
    ax.set_title("Top feature importance (gain)")
    savep = os.path.join(out_path, "feature_importance.png")
    fig.savefig(savep, bbox_inches="tight"); plt.close(fig)

def shap_analysis(model, X_train, X_test, feature_names, out_path, sample_n=1000):
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(n=min(sample_n, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    fig = shap.plots.bar(shap_values, max_display=30, show=False)
    plt.savefig(os.path.join(out_path, "shap_summary_bar.png"), bbox_inches="tight"); plt.close(fig)
    fig2 = shap.plots.beeswarm(shap_values, max_display=30, show=False)
    plt.savefig(os.path.join(out_path, "shap_beeswarm.png"), bbox_inches="tight"); plt.close(fig2)

# ------------------ Risk Stratification ------------------
def assign_risk(df_preds, method="threshold", threshold=10000.0, percentiles=(0.8,0.95)):
    out = df_preds.copy()
    if method == "threshold":
        out["Risk"] = pd.cut(out["predicted_cost"], bins=[-1, threshold, 3*threshold, out["predicted_cost"].max()+1], labels=["Low","Medium","High"])
    else:
        p1, p2 = percentiles
        q1 = out["predicted_cost"].quantile(p1)
        q2 = out["predicted_cost"].quantile(p2)
        out["Risk"] = pd.cut(out["predicted_cost"], bins=[-1, q1, q2, out["predicted_cost"].max()+1], labels=["Low","Medium","High"])
    return out

# ------------------ Main ------------------
def main(args):
    logger = setup_logger(args.output_dir)
    logger.info("Starting PMI predictive pipeline")
    membership_df = safe_read_csv(args.membership)
    claims_df = safe_read_csv(args.claims)

    members, claims = prepare_member_level(membership_df, claims_df, args.cutoff_date, horizon_months=args.horizon_months, windows_months=args.history_windows)
    members.to_csv(os.path.join(args.output_dir, "members_features.csv"), index=False)

    X_train, X_test, y_train, y_test, ids_train, ids_test, enc_map = preprocess_and_split(members, target_col="target_future_cost", test_size=args.test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    best_params = None
    if args.use_optuna:
        try:
            best_params = tune_lgb(X_tr, y_tr, X_val, y_val, n_trials=args.optuna_trials)
            logger.info(f"Optuna best params: {best_params}")
        except Exception as e:
            logger.warning("Optuna tuning failed: %s", e)
            best_params = None

    model = train_lgb_final(X_train, y_train, params=best_params)
    joblib.dump(model, os.path.join(args.output_dir, "model_lgb.pkl"))
    logger.info("Model trained and saved.")

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Test RMSE: {rmse:.2f}, R2: {r2:.3f}")

    preds_df = pd.DataFrame({"UniqueMemberRef": ids_test.values, "predicted_cost": y_pred, "true_cost": y_test})
    preds_df.to_csv(os.path.join(args.output_dir, "predictions_test.csv"), index=False)

    try:
        plot_feature_importance(model, X_train.columns.tolist(), args.output_dir)
        shap_analysis(model, X_train, X_test, X_train.columns.tolist(), args.output_dir, sample_n=args.shap_sample)
    except Exception as e:
        logger.warning("Failed to produce SHAP/importance: %s", e)

    if args.risk_method == "threshold":
        risk_df = assign_risk(preds_df, method="threshold", threshold=args.risk_threshold)
    else:
        risk_df = assign_risk(preds_df, method="percentile", percentiles=(args.pctile1, args.pctile2))
    risk_df.to_csv(os.path.join(args.output_dir, "risk_segments.csv"), index=False)
    logger.info("Risk segmentation saved. Pipeline completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PMI Predictive Pipeline")
    parser.add_argument("--membership", required=True, help="Membership CSV path")
    parser.add_argument("--claims", required=True, help="Claims CSV path")
    parser.add_argument("--output_dir", default="./pmi_output", help="Output folder")
    parser.add_argument("--cutoff_date", default="2023-01-01", help="Cutoff date for historical features (YYYY-MM-DD)")
    parser.add_argument("--horizon_months", type=int, default=24, help="Forecast horizon months")
    parser.add_argument("--history_windows", nargs="+", type=int, default=[6,12,24], help="History windows months for features")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--use_optuna", action="store_true", help="Use Optuna for tuning")
    parser.add_argument("--optuna_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--shap_sample", type=int, default=1000, help="SHAP sample size")
    parser.add_argument("--risk_method", choices=["threshold","percentile"], default="threshold", help="Risk segmentation method")
    parser.add_argument("--risk_threshold", type=float, default=10000.0, help="Threshold for high claims (GBP) when using threshold method")
    parser.add_argument("--pctile1", type=float, default=0.8, help="Lower percentile for percentile method (e.g., 0.8)")
    parser.add_argument("--pctile2", type=float, default=0.95, help="Upper percentile for percentile method (e.g., 0.95)")
    args = parser.parse_args()
    main(args)
