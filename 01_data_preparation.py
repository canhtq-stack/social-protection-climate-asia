"""
01_tabnet_causal.py
===================
Pipeline: XGBoost + TabNet (SHAP) → CausalForestDML (Baseline Estimation)

Paper : "Social Protection as Climate Adaptation: Heterogeneous Effects
         on Income Inequality in Asia"
Journal: Economics of Disasters and Climate Change, Springer

Outputs
-------
output/models/xgb_model.pkl
output/models/tabnet_model.*
output/models/causal_forest_baseline.pkl
output/tables/shap_xgb.csv
output/tables/shap_tabnet.csv
output/tables/shap_merged_table4.csv       → Table 4 in manuscript
output/tables/ate_summary.csv
output/tables/cate_individual.csv          → used by 04_blp_subsample.py
output/tables/cate_by_country.csv          → Table 2 / Figure 2
output/figures/figure3_shap_beeswarm.png   → Figure 3
output/figures/figure3b_shap_bar.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor
from econml.dml import CausalForestDML

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── Paths (relative to repo root) ─────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_FILE  = ROOT / "data" / "panel_qrei_final.csv"
MODEL_DIR  = ROOT / "output" / "models"
TABLE_DIR  = ROOT / "output" / "tables"
FIGURE_DIR = ROOT / "output" / "figures"

for d in [MODEL_DIR, TABLE_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model hyperparameters ──────────────────────────────────────────────────
N_FOLDS = 5

TABNET_CONFIG = dict(
    mask_type="entmax", n_d=32, n_a=32, n_steps=5, gamma=1.3,
    n_independent=2, n_shared=2, momentum=0.02, epsilon=1e-15,
    seed=RANDOM_SEED, verbose=0,
)
TABNET_FIT = dict(
    max_epochs=200, patience=50, batch_size=256,
    virtual_batch_size=128, num_workers=0,
)

# ── Variable definitions ───────────────────────────────────────────────────
Y_COL = "delta_gini"
T_COL = "high_social_prot"

# Effect modifiers X (forest splitting space; Table 4 SHAP)
X_COLS = [
    "temp_shock", "extreme_temp_shock", "rice_yield_dev",
    "log_gdp_pc", "rule_of_law", "democracy_electoral", "disaster_count_cy",
]
# Confounders W (partialled out by DML nuisance models)
W_COLS = ["log_gdp_pc_lag1", "gini_lag1", "corruption_index", "temp_shock_lag1"]

# Full feature set for SHAP auxiliary models (includes coverage for Table 4)
FEATURE_COLS = X_COLS + ["gini_lag1", "social_prot_coverage", "corruption_index"]

LABEL_MAP = {
    "rule_of_law":         "Rule of law",
    "democracy_electoral": "Electoral democracy",
    "log_gdp_pc":          "Log GDP per capita",
    "social_prot_coverage":"Social protection coverage",
    "corruption_index":    "Corruption index",
    "gini_lag1":           "Lagged Gini",
    "temp_shock":          "Temperature shock",
    "rice_yield_dev":      "Rice yield deviation",
    "extreme_temp_shock":  "Extreme temp. shock",
    "disaster_count_cy":   "Disaster count",
    "log_gdp_pc_lag1":     "Lagged log GDP per capita",
    "temp_shock_lag1":     "Lagged temp. anomaly",
}

# ── Helpers ────────────────────────────────────────────────────────────────

def section(title, step):
    print(f"\n{'='*70}\nSTEP {step}: {title}\n{'='*70}")

def available(cols, df):
    return [c for c in cols if c in df.columns]

def save_table(df, filename, label=""):
    path = TABLE_DIR / filename
    df.to_csv(path, index=True)
    print(f"  ✓ {filename}" + (f"  [{label}]" if label else ""))

def save_figure(fig, filename):
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {filename}")

# ── STEP 1: Load data ──────────────────────────────────────────────────────
section("LOAD DATA", 1)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}\n"
        "Place panel_qrei_final.csv in the data/ directory.\n"
        "See data/README_data.md for download instructions."
    )

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Create binary treatment if missing
if T_COL not in df.columns:
    if "social_prot_coverage" in df.columns:
        median_cov = df["social_prot_coverage"].median()
        df[T_COL] = (df["social_prot_coverage"].fillna(0) > median_cov).astype(float)
        print(f"  Created '{T_COL}' (threshold = {median_cov:.1f}%)")
    else:
        raise KeyError(f"Column '{T_COL}' missing and 'social_prot_coverage' not found.")

feat_cols = available(FEATURE_COLS, df)
x_cols    = available(X_COLS, df)
w_cols    = available(W_COLS, df)

df_model = df[df[Y_COL].notna() & df[T_COL].notna()].copy()
print(f"  Estimation sample: {len(df_model):,} obs")

# Impute missing values (country median → global median)
for col in set(feat_cols + x_cols + w_cols):
    if col in df_model.columns and df_model[col].isnull().any():
        df_model[col] = (
            df_model.groupby("country_code")[col]
            .transform(lambda x: x.fillna(x.median()))
        )
        df_model[col] = df_model[col].fillna(df_model[col].median())

X_all    = df_model[feat_cols].values.astype(np.float32)
y_all    = df_model[Y_COL].values.astype(np.float32)
T_all    = df_model[T_COL].values.astype(np.float32)
groups   = df_model["country_code"].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

print(f"  Feature matrix: {X_all.shape}  |  Treated share: {T_all.mean():.3f}")

# ── STEP 2: XGBoost + SHAP ────────────────────────────────────────────────
section("XGBoost — Train + SHAP", 2)

xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
)
xgb_model.fit(X_scaled, y_all)

gkf  = GroupKFold(n_splits=N_FOLDS)
cv_r2 = cross_val_score(
    xgb_model, X_scaled, y_all,
    cv=gkf.split(X_scaled, y_all, groups), scoring="r2",
)
print(f"  CV R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

joblib.dump(xgb_model, MODEL_DIR / "xgb_model.pkl")

explainer_xgb = shap.TreeExplainer(xgb_model)
shap_vals_xgb = explainer_xgb.shap_values(X_scaled)
mean_shap_xgb = np.abs(shap_vals_xgb).mean(axis=0)

df_shap_xgb = pd.DataFrame({
    "Feature":       feat_cols,
    "XGB_mean_shap": mean_shap_xgb,
    "XGB_rank":      pd.Series(mean_shap_xgb).rank(ascending=False).astype(int).values,
}).sort_values("XGB_mean_shap", ascending=False)

save_table(df_shap_xgb.set_index("Feature"), "shap_xgb.csv", "XGBoost SHAP")

# ── STEP 3: TabNet + SHAP ─────────────────────────────────────────────────
section("TabNet — Train + SHAP", 3)

X_tab = X_scaled.astype(np.float32)
y_tab = y_all.reshape(-1, 1).astype(np.float32)

n_val   = max(int(len(X_tab) * 0.2), 10)
n_train = len(X_tab) - n_val

tabnet = TabNetRegressor(**TABNET_CONFIG)
tabnet.fit(
    X_train=X_tab[:n_train], y_train=y_tab[:n_train],
    eval_set=[(X_tab[n_train:], y_tab[n_train:])],
    eval_name=["val"], eval_metric=["rmse"],
    **TABNET_FIT,
)

y_pred_tab = tabnet.predict(X_tab).ravel()
print(f"  TabNet R² = {r2_score(y_all, y_pred_tab):.4f} | "
      f"RMSE = {np.sqrt(mean_squared_error(y_all, y_pred_tab)):.4f}")

tabnet.save_model(str(MODEL_DIR / "tabnet_model"))

print("  Computing TabNet SHAP (KernelExplainer — this may take several minutes)...")
background    = shap.sample(X_tab, 50, random_state=RANDOM_SEED)
explainer_tab = shap.KernelExplainer(
    lambda x: tabnet.predict(x.astype(np.float32)).ravel(), background,
)
shap_vals_tab = explainer_tab.shap_values(X_tab, nsamples=100, l1_reg="num_features(5)")
mean_shap_tab = np.abs(shap_vals_tab).mean(axis=0)

df_shap_tab = pd.DataFrame({
    "Feature":       feat_cols,
    "TAB_mean_shap": mean_shap_tab,
    "TAB_rank":      pd.Series(mean_shap_tab).rank(ascending=False).astype(int).values,
}).sort_values("TAB_mean_shap", ascending=False)

save_table(df_shap_tab.set_index("Feature"), "shap_tabnet.csv", "TabNet SHAP")

# ── STEP 4: Merge SHAP → Table 4 ─────────────────────────────────────────
section("Merged SHAP Table — Table 4", 4)

df_shap_merged = df_shap_xgb.merge(
    df_shap_tab[["Feature", "TAB_mean_shap", "TAB_rank"]],
    on="Feature", how="outer",
).sort_values("XGB_mean_shap", ascending=False)

df_shap_merged["Label"] = (
    df_shap_merged["Feature"].map(LABEL_MAP)
    .fillna(df_shap_merged["Feature"])
)
print(df_shap_merged[["Label", "XGB_mean_shap", "XGB_rank",
                        "TAB_mean_shap", "TAB_rank"]].to_string(index=False))
save_table(df_shap_merged.set_index("Feature"), "shap_merged_table4.csv", "Table 4")

# ── STEP 5: Figure 3 ──────────────────────────────────────────────────────
section("Figure 3 — SHAP Beeswarm + Bar", 5)

feat_order = df_shap_xgb["Feature"].tolist()
feat_idx   = [feat_cols.index(f) for f in feat_order if f in feat_cols]
feat_labels = [LABEL_MAP.get(f, f) for f in feat_order if f in feat_cols]

fig3a, _ = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_vals_xgb[:, feat_idx], X_scaled[:, feat_idx],
    feature_names=feat_labels, plot_type="dot", show=False,
    color_bar=True, max_display=len(feat_labels), plot_size=None,
)
plt.gca().set_title(
    f"Figure 3. SHAP Beeswarm — XGBoost  (N = {len(X_scaled)})\n"
    "Blue = low feature value → red = high. "
    "Positive SHAP = association with inequality increase.",
    fontsize=9, pad=10,
)
plt.tight_layout()
save_figure(fig3a, "figure3_shap_beeswarm.png")

top_n  = min(10, len(df_shap_merged))
df_top = df_shap_merged.head(top_n).copy()
x_pos  = np.arange(len(df_top))
width  = 0.38

fig3b, ax = plt.subplots(figsize=(11, 6))
ax.barh(x_pos + width / 2, df_top["XGB_mean_shap"], width,
        color="#2B5FA5", alpha=0.85, label="XGBoost")
ax.barh(x_pos - width / 2, df_top["TAB_mean_shap"], width,
        color="#E07B39", alpha=0.85, label="TabNet")
ax.set_yticks(x_pos)
ax.set_yticklabels(df_top["Label"].tolist(), fontsize=9)
ax.set_xlabel("Mean |SHAP| Value")
ax.set_title(
    f"Figure 3 (panel B). SHAP Feature Importance — XGBoost vs TabNet\n"
    f"N = {len(X_scaled)}"
)
ax.legend(fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.grid(axis="x", alpha=0.25, linestyle=":")
plt.tight_layout()
save_figure(fig3b, "figure3b_shap_bar.png")

# ── STEP 6: CausalForestDML — Baseline ────────────────────────────────────
section("CausalForestDML — Baseline Estimation", 6)

X_cf = df_model[x_cols].values.astype(np.float64)
W_cf = df_model[w_cols].values.astype(np.float64) if w_cols else None
Y_cf = df_model[Y_COL].values.astype(np.float64)
T_cf = (df_model[T_COL].values >= 0.5).astype(int)

print(f"  N = {len(Y_cf)}"
      f"  |  X: {X_cf.shape[1]} cols"
      f"  |  W: {W_cf.shape[1] if W_cf is not None else 0} cols"
      f"  |  Treated: {T_cf.sum()} ({T_cf.mean():.1%})")

cf = CausalForestDML(
    model_y=RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=-1,
    ),
    model_t=RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=-1,
    ),
    discrete_treatment=True,
    n_estimators=500,
    min_samples_leaf=10,
    max_samples=0.5,
    inference=True,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
cf.fit(Y=Y_cf, T=T_cf, X=X_cf, W=W_cf)

ate   = float(cf.ate(X_cf))
ci    = cf.ate_interval(X_cf, alpha=0.05)
ci_lo, ci_hi = float(ci[0]), float(ci[1])
print(f"  ATE = {ate:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

joblib.dump(cf, MODEL_DIR / "causal_forest_baseline.pkl")

save_table(
    pd.DataFrame([{
        "ATE": ate, "CI_lower": ci_lo, "CI_upper": ci_hi,
        "N_obs": len(Y_cf), "N_treated": int(T_cf.sum()),
        "pct_treated": round(T_cf.mean(), 4),
    }]),
    "ate_summary.csv", "Full-sample ATE",
)

# Individual CATEs (required by 04_blp_subsample.py and 06_blp_cluster_bootstrap.py)
cate_vals = cf.effect(X_cf)
id_cols   = [c for c in ["country_code", "year"] if c in df_model.columns]
df_cate   = df_model[id_cols].copy().reset_index(drop=True)
df_cate["cate"] = cate_vals
for i, col in enumerate(x_cols):
    df_cate[col] = X_cf[:, i]
save_table(df_cate, "cate_individual.csv", "Individual CATEs → used by 04, 06")

# Country-level CATEs (Table 2 / Figure 2)
df_cate["_cc"] = df_model["country_code"].values
cate_by_country = (
    df_cate.groupby("_cc")
    .apply(lambda g: pd.Series({
        "cate_mean": g["cate"].mean(),
        "cate_lb":   np.percentile(g["cate"], 2.5),
        "cate_ub":   np.percentile(g["cate"], 97.5),
        "n_obs":     len(g),
    }))
    .reset_index()
    .rename(columns={"_cc": "country_code"})
)
save_table(
    cate_by_country.set_index("country_code"),
    "cate_by_country.csv", "Country-level CATEs → Table 2 / Figure 2",
)

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("DONE — 01_tabnet_causal.py")
print(f"  Next step: python code/02_policy_sim.py")
print("="*70)
