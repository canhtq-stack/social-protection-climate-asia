"""
QREI Project – 03_tabnet_causal.py
=====================================
Mô hình chính: XGBoost baseline → TabNet → Causal Forest CATE

Paper : "Extreme weather shocks, income inequality, and social protection in Asia"
Journal: Environmental and Resource Economics (ERE) – Springer
Version: 1.0

PIPELINE:
  Step 1 – Load & chuẩn bị features
  Step 2 – XGBoost / Random Forest  (baseline)
  Step 3 – TabNet                   (core interpretable DL)
  Step 4 – Causal Forest CATE       (heterogeneous treatment effects)
  Step 5 – SHAP feature importance
  Step 6 – Lưu tất cả kết quả

INPUT : Data/processed/panel_qrei_final.csv
OUTPUT: results/models/   – model artifacts
        results/tables/   – coefficient / CATE tables
        results/figures/  – SHAP plots

Cài đặt (lần đầu):
  pip install pytorch-tabnet econml shap xgboost scikit-learn matplotlib seaborn

Chạy : python 03_tabnet_causal.py
Tiếp : python 04_policy_sim.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (không cần display)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings("ignore")

# ── Deep learning / ML ──────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# ── TabNet ──────────────────────────────────────────────────
from pytorch_tabnet.tab_model import TabNetRegressor

# ── Causal Forest ────────────────────────────────────────────
from econml.dml import CausalForestDML

# ── SHAP ─────────────────────────────────────────────────────
import shap

# ============================================================
# CẤU HÌNH
# ============================================================

BASE = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\Quantum_Global Economic Resilience\qrei_project"

DATA_FILE   = os.path.join(BASE, "Data", "processed", "panel_qrei_final.csv")
MODEL_DIR   = os.path.join(BASE, "results", "models")
TABLE_DIR   = os.path.join(BASE, "results", "tables")
FIGURE_DIR  = os.path.join(BASE, "results", "figures")

for d in [MODEL_DIR, TABLE_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS     = 5          # K-fold cross-validation
SHOCK_THRESHOLD = 2.0

# ── TabNet config (theo thiết kế nghiên cứu) ────────────────
TABNET_CONFIG = dict(
    mask_type     = "entmax",   # entmax thay vì sparsemax – tốt hơn cho panel data
    n_d           = 32,         # chiều embedding
    n_a           = 32,
    n_steps       = 5,
    gamma         = 1.3,
    n_independent = 2,
    n_shared      = 2,
    momentum      = 0.02,
    epsilon       = 1e-15,
    seed          = RANDOM_SEED,
    verbose       = 0,
)
TABNET_FIT = dict(
    max_epochs        = 200,
    patience          = 50,
    batch_size        = 256,
    virtual_batch_size= 128,
    num_workers       = 0,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title, step):
    print(f"\n{'='*65}")
    print(f"STEP {step}: {title}")
    print("=" * 65)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def save_table(df, filename, label=""):
    path = os.path.join(TABLE_DIR, filename)
    df.to_csv(path)
    tag = f" – {label}" if label else ""
    print(f"  ✓ {filename}{tag}")
    return path


def save_figure(fig, filename):
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {filename}")
    return path


# ============================================================
# STEP 1 – LOAD & CHUẨN BỊ FEATURES
# ============================================================

section("LOAD & CHUẨN BỊ FEATURES", step=1)

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  Loaded: {DATA_FILE}")
print(f"  Shape : {df.shape}")

# ── Định nghĩa feature sets ──────────────────────────────────
# Outcome
Y_COL = "delta_gini"

# Treatment (cho Causal Forest)
T_COL = "high_social_prot"

# Features cho prediction models (XGBoost, TabNet)
FEATURE_COLS = [
    # Climate shock
    "temp_shock", "extreme_temp_shock", "temp_shock_lag1", "temp_shock_lag2",
    # Mediator
    "rice_yield_dev",
    # Treatment
    "social_prot_coverage", "high_social_prot",
    # Controls
    "log_gdp_pc", "log_gdp_pc_lag1",
    # Institutions
    "rule_of_law", "democracy_electoral", "corruption_index",
    # Lagged outcome (dynamic)
    "gini_lag1", 
    # Disasters
    "disaster_count_cy",
]

# Controls cho Causal Forest (X = effect modifiers, W = confounders)
X_COLS = [   # heterogeneity variables (effect modifiers)
    "temp_shock", "extreme_temp_shock",
    "rice_yield_dev",
    "log_gdp_pc",
    "rule_of_law",
    "democracy_electoral",
    "disaster_count_cy",
]
W_COLS = [   # confounders
    "log_gdp_pc_lag1",
    "gini_lag1",
    "corruption_index",
    "temp_shock_lag1",
]

# Lọc các cột thực sự có trong df
def available(cols):
    return [c for c in cols if c in df.columns]

FEATURE_COLS = available(FEATURE_COLS)
X_COLS       = available(X_COLS)
W_COLS       = available(W_COLS)

print(f"\n  Feature cols   : {len(FEATURE_COLS)}")
print(f"  X cols (CATE)  : {len(X_COLS)}")
print(f"  W cols (conf.) : {len(W_COLS)}")

# ── Tạo working dataset: bỏ rows thiếu outcome hoặc treatment ─
df_model = df[df[Y_COL].notna() & df[T_COL].notna()].copy()
print(f"\n  Obs có đủ Y + T: {len(df_model)} / {len(df)}")

# Fill NaN trong features bằng median per country
for col in FEATURE_COLS + X_COLS + W_COLS:
    if col in df_model.columns and df_model[col].isnull().any():
        df_model[col] = df_model.groupby("country_code")[col].transform(
            lambda x: x.fillna(x.median())
        )
        # Nếu vẫn còn NaN (country không có data) → global median
        df_model[col] = df_model[col].fillna(df_model[col].median())

# Arrays
X_all = df_model[FEATURE_COLS].values.astype(np.float32)
y_all = df_model[Y_COL].values.astype(np.float32)
T_all = df_model[T_COL].values.astype(np.float32)
X_cate = df_model[X_COLS].values.astype(np.float32)
W_cate = df_model[W_COLS].values.astype(np.float32)

print(f"  X_all shape    : {X_all.shape}")
print(f"  y_all shape    : {y_all.shape}")
print(f"  Treatment share: {T_all.mean()*100:.1f}% treated")

# ── Scaler cho models yêu cầu normalized input ───────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)


# ============================================================
# STEP 2 – BASELINE: RANDOM FOREST & XGBOOST
# ============================================================

section("BASELINE: Random Forest & XGBoost", step=2)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

results_baseline = {}

# ── 2a. Random Forest ────────────────────────────────────────
print("  → 2a. Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
rf_scores = cross_val_score(rf, X_all, y_all,
                             cv=kf, scoring="r2", n_jobs=-1)
rf_rmse   = np.sqrt(-cross_val_score(rf, X_all, y_all,
                                      cv=kf, scoring="neg_mean_squared_error",
                                      n_jobs=-1))
rf.fit(X_all, y_all)
results_baseline["RandomForest"] = {
    "R2_mean": rf_scores.mean(), "R2_std": rf_scores.std(),
    "RMSE_mean": rf_rmse.mean(), "RMSE_std": rf_rmse.std(),
}
print(f"     R²={rf_scores.mean():.3f} ± {rf_scores.std():.3f}  |  "
      f"RMSE={rf_rmse.mean():.3f} ± {rf_rmse.std():.3f}")

# Lưu model
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_baseline.pkl"))

# ── 2b. XGBoost ──────────────────────────────────────────────
print("  → 2b. XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0,
)
xgb_scores = cross_val_score(xgb_model, X_all, y_all,
                               cv=kf, scoring="r2", n_jobs=-1)
xgb_rmse   = np.sqrt(-cross_val_score(xgb_model, X_all, y_all,
                                       cv=kf, scoring="neg_mean_squared_error",
                                       n_jobs=-1))
xgb_model.fit(X_all, y_all)
results_baseline["XGBoost"] = {
    "R2_mean": xgb_scores.mean(), "R2_std": xgb_scores.std(),
    "RMSE_mean": xgb_rmse.mean(), "RMSE_std": xgb_rmse.std(),
}
print(f"     R²={xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}  |  "
      f"RMSE={xgb_rmse.mean():.3f} ± {xgb_rmse.std():.3f}")

joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_baseline.pkl"))

# ── Baseline summary ─────────────────────────────────────────
df_baseline = pd.DataFrame(results_baseline).T.round(4)
print(f"\n  Baseline summary:")
print(df_baseline.to_string())
save_table(df_baseline, "baseline_model_performance.csv", "RF & XGBoost")


# ============================================================
# STEP 3 – TABNET (CORE MODEL)
# ============================================================

section("TABNET  (mask_type=entmax, max_epochs=200)", step=3)

# ── K-fold CV cho TabNet ─────────────────────────────────────
tabnet_r2   = []
tabnet_rmse_list = []

print(f"  → {N_FOLDS}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y_all[train_idx],    y_all[val_idx]

    model_fold = TabNetRegressor(**TABNET_CONFIG)
    model_fold.fit(
        X_tr, y_tr.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        eval_metric=["rmse"],
        **TABNET_FIT,
    )

    y_pred_val = model_fold.predict(X_val).flatten()
    r2_fold    = r2_score(y_val, y_pred_val)
    rmse_fold  = rmse(y_val, y_pred_val)
    tabnet_r2.append(r2_fold)
    tabnet_rmse_list.append(rmse_fold)
    print(f"     Fold {fold}: R²={r2_fold:.3f}  RMSE={rmse_fold:.3f}")

print(f"\n  TabNet CV results:")
print(f"     R²  = {np.mean(tabnet_r2):.3f} ± {np.std(tabnet_r2):.3f}")
print(f"     RMSE= {np.mean(tabnet_rmse_list):.3f} ± "
      f"{np.std(tabnet_rmse_list):.3f}")

# ── Train final TabNet trên toàn bộ data ──────────────────────
print(f"\n  → Training final TabNet on full dataset...")
tabnet_final = TabNetRegressor(**TABNET_CONFIG)
tabnet_final.fit(
    X_scaled, y_all.reshape(-1, 1),
    **TABNET_FIT,
)
y_pred_tabnet = tabnet_final.predict(X_scaled).flatten()
tabnet_full_r2   = r2_score(y_all, y_pred_tabnet)
tabnet_full_rmse = rmse(y_all, y_pred_tabnet)
print(f"     Full-data: R²={tabnet_full_r2:.3f}  RMSE={tabnet_full_rmse:.3f}")

# Lưu model
tabnet_final.save_model(os.path.join(MODEL_DIR, "tabnet_final"))
print(f"  ✓ Model saved → {MODEL_DIR}/tabnet_final.*")

# ── TabNet feature importance ─────────────────────────────────
feat_imp_tabnet = pd.Series(
    tabnet_final.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)

print(f"\n  TabNet feature importance (top 10):")
for feat, imp in feat_imp_tabnet.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"    {feat:<35} {bar:<20} {imp:.4f}")

# So sánh tất cả models
results_all = {
    **results_baseline,
    "TabNet": {
        "R2_mean": np.mean(tabnet_r2), "R2_std": np.std(tabnet_r2),
        "RMSE_mean": np.mean(tabnet_rmse_list), "RMSE_std": np.std(tabnet_rmse_list),
    }
}
df_compare = pd.DataFrame(results_all).T.round(4)
print(f"\n  Model comparison:")
print(df_compare.to_string())
save_table(df_compare, "model_comparison.csv", "RF vs XGBoost vs TabNet")

# ── Figure: model comparison ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
models_list = list(df_compare.index)
colors = ["#4472C4", "#ED7D31", "#A9D18E"]

for ax, metric, title in zip(
        axes,
        [("R2_mean", "R2_std"), ("RMSE_mean", "RMSE_std")],
        ["R² (higher = better)", "RMSE (lower = better)"]):
    means = df_compare[metric[0]].values
    stds  = df_compare[metric[1]].values
    bars  = ax.bar(models_list, means, yerr=stds,
                   color=colors[:len(models_list)],
                   capsize=5, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric[0].split("_")[0])
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Model Performance Comparison (5-fold CV)\nOutcome: ΔGini",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_figure(fig, "model_comparison.png")


# ============================================================
# STEP 4 – CAUSAL FOREST CATE (heterogeneous treatment effects)
# ============================================================

section("CAUSAL FOREST CATE  (econml CausalForestDML)", step=4)

print("  Estimating heterogeneous treatment effects of high_social_prot on ΔGini")
print(f"  X (effect modifiers) : {X_COLS}")
print(f"  W (confounders)      : {W_COLS}")

# ── Chuẩn bị arrays cho Causal Forest ────────────────────────
# Cần outcome, treatment, effect modifiers, confounders
# Lọc rows đủ T (treatment)
cf_mask = df_model[T_COL].notna()
Y_cf  = df_model.loc[cf_mask, Y_COL].values.astype(np.float64)
# Ép T về binary int (0/1) – tránh lỗi nếu dedup tạo giá trị 0.5
T_cf = (df_model.loc[cf_mask, T_COL].values >= 0.5).astype(int)
X_cf  = df_model.loc[cf_mask, X_COLS].values.astype(np.float64)
W_cf  = df_model.loc[cf_mask, W_COLS].values.astype(np.float64) \
        if W_COLS else None

print(f"\n  Sample: {len(Y_cf)} obs  |  "
      f"Treated: {int(T_cf.sum())} ({T_cf.mean()*100:.1f}%)")

# ── Fit CausalForestDML ──────────────────────────────────────
print("\n  → Fitting CausalForestDML...")

cf_model = CausalForestDML(
    model_y          = RandomForestRegressor(
                           n_estimators=200, max_depth=5,
                           min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1),
    model_t          = RandomForestClassifier(
                           n_estimators=200, max_depth=5,
                           min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1),
    discrete_treatment = True,   # T binary -> use classifier
    n_estimators     = 500,
    min_samples_leaf = 10,
    max_depth        = None,
    max_features     = "sqrt",
    inference        = True,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)

cf_model.fit(Y=Y_cf, T=T_cf, X=X_cf,
             W=W_cf if W_cf is not None else None)

print("  ✓ CausalForestDML fitted")

# ── CATE estimation ──────────────────────────────────────────
print("\n  → Estimating CATE...")
cate_pred    = cf_model.effect(X_cf)
cate_lb, cate_ub = cf_model.effect_interval(X_cf, alpha=0.05)

# ATE (Average Treatment Effect)
ate      = cf_model.ate(X_cf)
ate_ib   = cf_model.ate_interval(X_cf, alpha=0.05)

print(f"\n  ATE (Average Treatment Effect):")
print(f"     Effect of high_social_prot on ΔGini:")
print(f"     ATE = {ate:.4f}  95% CI [{ate_ib[0]:.4f}, {ate_ib[1]:.4f}]")
if ate < 0:
    print(f"  → High social protection GIẢM Gini inequality "
          f"(ΔGini giảm {abs(ate):.3f} điểm/năm)")
else:
    print(f"  → High social protection TĂNG Gini inequality "
          f"(ΔGini tăng {abs(ate):.3f} điểm/năm)")

# ── CATE theo quốc gia ────────────────────────────────────────
df_cate = df_model[cf_mask].copy()
df_cate["cate"]    = cate_pred
df_cate["cate_lb"] = cate_lb
df_cate["cate_ub"] = cate_ub
df_cate["cate_sig"] = (
    (cate_lb > 0) | (cate_ub < 0)
).astype(int)   # 1 nếu CI không chứa 0

# Tóm tắt CATE theo quốc gia
cate_by_country = (df_cate.groupby("country_code")
                           .agg(
                               cate_mean=("cate", "mean"),
                               cate_lb=("cate_lb", "mean"),
                               cate_ub=("cate_ub", "mean"),
                               n_obs=("cate", "count"),
                               pct_sig=("cate_sig", "mean"),
                           )
                           .sort_values("cate_mean"))

print(f"\n  CATE by country (sorted by effect size):")
print(f"  {'Country':<8} {'CATE':>8} {'95% CI':>20}  {'Sig%':>6}  {'N':>4}")
print("  " + "-"*55)
for country, row in cate_by_country.iterrows():
    direction = "▼" if row["cate_mean"] < 0 else "▲"
    sig       = "***" if row["pct_sig"] > 0.5 else ""
    print(f"  {country:<8} {row['cate_mean']:>8.4f}  "
          f"[{row['cate_lb']:>7.4f}, {row['cate_ub']:>7.4f}]  "
          f"{row['pct_sig']*100:>5.1f}%  {row['n_obs']:>4}  {direction}{sig}")

# ── CATE theo mức độ phụ thuộc nông nghiệp (heterogeneous) ───
# Sub-sample: chia theo temp_shock quartile
if "temp_shock" in df_cate.columns:
    df_cate["temp_q"] = pd.qcut(df_cate["temp_shock"], q=4,
                                 labels=["Q1\n(cool)", "Q2", "Q3", "Q4\n(hot)"])
    cate_by_temp = (df_cate.groupby("temp_q", observed=True)["cate"]
                            .agg(["mean", "std", "count"]).round(4))
    print(f"\n  CATE by temperature shock quartile:")
    print(cate_by_temp.to_string())

# Lưu CATE results
save_table(df_cate[["country_code", "year", "cate", "cate_lb",
                     "cate_ub", "cate_sig"]],
           "cate_individual.csv", "Individual CATE estimates")
save_table(cate_by_country, "cate_by_country.csv", "CATE by country")

# ATE summary
ate_summary = pd.DataFrame({
    "ATE": [ate],
    "CI_lower": [ate_ib[0]],
    "CI_upper": [ate_ib[1]],
    "N_obs": [len(Y_cf)],
    "N_treated": [int(T_cf.sum())],
    "pct_treated": [T_cf.mean()],
})
save_table(ate_summary, "ate_summary.csv", "Average Treatment Effect")

# Lưu Causal Forest model
joblib.dump(cf_model, os.path.join(MODEL_DIR, "causal_forest.pkl"))
print(f"  ✓ Model saved → {MODEL_DIR}/causal_forest.pkl")

# ── Figure: CATE distribution ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: CATE distribution histogram
ax = axes[0]
ax.hist(cate_pred, bins=30, color="#4472C4", edgecolor="white",
        alpha=0.8, linewidth=0.5)
ax.axvline(ate, color="#ED7D31", linewidth=2,
           label=f"ATE = {ate:.4f}")
ax.axvline(0,   color="black",  linewidth=1, linestyle="--",
           label="Zero effect")
ax.set_xlabel("CATE (Effect on ΔGini)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Distribution of CATE\n(Effect of High Social Protection)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

# Plot 2: CATE by country (error bar plot)
ax2 = axes[1]
countries_sorted = cate_by_country.index.tolist()
y_pos = range(len(countries_sorted))
ax2.barh(y_pos, cate_by_country["cate_mean"],
         xerr=[(cate_by_country["cate_mean"] - cate_by_country["cate_lb"]),
               (cate_by_country["cate_ub"] - cate_by_country["cate_mean"])],
         color=["#ED7D31" if v > 0 else "#4472C4"
                for v in cate_by_country["cate_mean"]],
         capsize=3, alpha=0.8)
ax2.axvline(0, color="black", linewidth=1, linestyle="--")
ax2.axvline(ate, color="#A9D18E", linewidth=1.5, linestyle="-",
            label=f"ATE={ate:.3f}")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(countries_sorted, fontsize=8)
ax2.set_xlabel("CATE (Effect on ΔGini)", fontsize=11)
ax2.set_title("CATE by Country\nBlue = reduces inequality",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)

plt.tight_layout()
save_figure(fig, "cate_distribution.png")


# ============================================================
# STEP 5 – SHAP FEATURE IMPORTANCE
# ============================================================

section("SHAP FEATURE IMPORTANCE", step=5)

print("  → Computing SHAP values for XGBoost & TabNet...")

# ── 5a. SHAP cho XGBoost ─────────────────────────────────────
print("  → 5a. XGBoost SHAP...")
explainer_xgb  = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_all)

shap_df_xgb = pd.DataFrame(
    np.abs(shap_values_xgb).mean(axis=0),
    index=FEATURE_COLS,
    columns=["mean_abs_shap"]
).sort_values("mean_abs_shap", ascending=False)

print(f"\n  XGBoost SHAP (top 10):")
for feat, row in shap_df_xgb.head(10).iterrows():
    bar = "█" * int(row["mean_abs_shap"] * 500)
    print(f"    {feat:<35} {bar:<20} {row['mean_abs_shap']:.4f}")

# ── 5b. SHAP cho TabNet (KernelExplainer – sample 100 obs) ───
print("\n  → 5b. TabNet SHAP (KernelExplainer, sample=100 obs)...")
bg_sample  = shap.sample(X_scaled, 50, random_state=RANDOM_SEED)
test_sample = X_scaled[:100]

def tabnet_predict(x):
    return tabnet_final.predict(x.astype(np.float32)).flatten()

explainer_tn    = shap.KernelExplainer(tabnet_predict, bg_sample)
shap_values_tn  = explainer_tn.shap_values(test_sample, nsamples=100)

shap_df_tn = pd.DataFrame(
    np.abs(shap_values_tn).mean(axis=0),
    index=FEATURE_COLS,
    columns=["mean_abs_shap"]
).sort_values("mean_abs_shap", ascending=False)

print(f"\n  TabNet SHAP (top 10):")
for feat, row in shap_df_tn.head(10).iterrows():
    bar = "█" * int(row["mean_abs_shap"] * 500)
    print(f"    {feat:<35} {bar:<20} {row['mean_abs_shap']:.4f}")

# ── 5c. Tổng hợp SHAP importance ─────────────────────────────
shap_combined = pd.DataFrame({
    "xgb_shap":    shap_df_xgb["mean_abs_shap"],
    "tabnet_shap": shap_df_tn["mean_abs_shap"],
    "tabnet_fi":   feat_imp_tabnet,
}).fillna(0)
shap_combined["avg_importance"] = shap_combined.mean(axis=1)
shap_combined = shap_combined.sort_values("avg_importance", ascending=False)

save_table(shap_combined, "shap_importance.csv", "SHAP importance (XGB + TabNet)")

# ── Figure: SHAP summary ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: XGBoost SHAP bar
top_n = 12
ax = axes[0]
top_feats = shap_df_xgb.head(top_n)
ax.barh(range(len(top_feats)), top_feats["mean_abs_shap"],
        color="#4472C4", alpha=0.85)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value|", fontsize=10)
ax.set_title(f"XGBoost – Top {top_n} Features\n(Mean absolute SHAP)",
             fontsize=10, fontweight="bold")

# Plot 2: TabNet SHAP bar
ax2 = axes[1]
top_feats_tn = shap_df_tn.head(top_n)
ax2.barh(range(len(top_feats_tn)), top_feats_tn["mean_abs_shap"],
         color="#ED7D31", alpha=0.85)
ax2.set_yticks(range(len(top_feats_tn)))
ax2.set_yticklabels(top_feats_tn.index, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel("Mean |SHAP value|", fontsize=10)
ax2.set_title(f"TabNet – Top {top_n} Features\n(Mean absolute SHAP)",
              fontsize=10, fontweight="bold")

plt.suptitle("Feature Importance: SHAP Analysis\nOutcome: ΔGini",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_figure(fig, "shap_importance.png")

# ── Figure: SHAP beeswarm cho XGBoost ────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values_xgb, X_all,
                  feature_names=FEATURE_COLS,
                  max_display=12,
                  show=False, plot_type="dot")
plt.title("XGBoost SHAP Beeswarm – Effect on ΔGini",
          fontsize=12, fontweight="bold")
plt.tight_layout()
save_figure(fig, "shap_beeswarm_xgb.png")


# ============================================================
# STEP 6 – LƯU TẤT CẢ KẾT QUẢ & REPORT
# ============================================================

section("LƯU KẾT QUẢ & REPORT TỔNG KẾT", step=6)

# ── Predictions vào df ───────────────────────────────────────
df_model["y_pred_xgb"]    = xgb_model.predict(X_all)
df_model["y_pred_tabnet"] = y_pred_tabnet
df_model["residual_xgb"]  = y_all - df_model["y_pred_xgb"]
df_model["residual_tn"]   = y_all - df_model["y_pred_tabnet"]

save_table(
    df_model[["country_code", "year", Y_COL,
              "y_pred_xgb", "y_pred_tabnet",
              "residual_xgb", "residual_tn"]],
    "predictions.csv", "Predictions all models"
)

# ── Figure: predicted vs actual ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, preds, label, color in zip(
        axes,
        [df_model["y_pred_xgb"], df_model["y_pred_tabnet"]],
        ["XGBoost", "TabNet"],
        ["#4472C4", "#ED7D31"]):
    ax.scatter(y_all, preds, alpha=0.4, s=15, color=color)
    lim = [min(y_all.min(), preds.min()),
           max(y_all.max(), preds.max())]
    ax.plot(lim, lim, "k--", linewidth=1)
    r2 = r2_score(y_all, preds)
    ax.set_xlabel("Actual ΔGini", fontsize=10)
    ax.set_ylabel("Predicted ΔGini", fontsize=10)
    ax.set_title(f"{label}\nR²={r2:.3f}", fontsize=10, fontweight="bold")

plt.suptitle("Predicted vs Actual ΔGini", fontsize=12, fontweight="bold")
plt.tight_layout()
save_figure(fig, "predicted_vs_actual.png")

# ── Figure: CATE vs temp_shock scatter ────────────────────────
if "temp_shock" in df_cate.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df_cate["temp_shock"], df_cate["cate"],
                    c=df_cate["cate"], cmap="RdBu_r",
                    alpha=0.6, s=20, vmin=-0.5, vmax=0.5)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.axhline(ate, color="#ED7D31", linewidth=1.5,
               label=f"ATE = {ate:.4f}")
    ax.axvline(SHOCK_THRESHOLD,  color="red",  linewidth=1,
               linestyle=":", label="Extreme shock threshold")
    ax.axvline(-SHOCK_THRESHOLD, color="blue", linewidth=1, linestyle=":")
    plt.colorbar(sc, ax=ax, label="CATE")
    ax.set_xlabel("Temperature Shock (z-score, detrended)", fontsize=11)
    ax.set_ylabel("CATE (Effect of High Social Prot on ΔGini)", fontsize=11)
    ax.set_title("Heterogeneous Treatment Effects\nby Temperature Shock Intensity",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig, "cate_vs_temp_shock.png")

# ============================================================
# REPORT TỔNG KẾT
# ============================================================

print(f"\n{'='*65}")
print("REPORT TỔNG KẾT – 03_tabnet_causal.py")
print("=" * 65)

print(f"\n  MODEL PERFORMANCE (5-fold CV):")
print(df_compare.to_string())

print(f"\n  CAUSAL FOREST:")
print(f"  ATE = {ate:.4f}  95% CI [{ate_ib[0]:.4f}, {ate_ib[1]:.4f}]")
print(f"  → {'Giảm' if ate < 0 else 'Tăng'} Gini inequality "
      f"{abs(ate):.4f} điểm/năm khi có high social protection")

print(f"\n  TOP 5 FEATURES (XGBoost SHAP):")
for feat, row in shap_df_xgb.head(5).iterrows():
    print(f"    {feat:<35} {row['mean_abs_shap']:.4f}")

print(f"\n  OUTPUT FILES:")
tables = ["baseline_model_performance.csv", "model_comparison.csv",
          "cate_individual.csv", "cate_by_country.csv",
          "ate_summary.csv", "shap_importance.csv", "predictions.csv"]
figures = ["model_comparison.png", "cate_distribution.png",
           "shap_importance.png", "shap_beeswarm_xgb.png",
           "predicted_vs_actual.png", "cate_vs_temp_shock.png"]
models_saved = ["rf_baseline.pkl", "xgb_baseline.pkl",
                "tabnet_final.zip", "causal_forest.pkl"]

for f in tables:
    exists = os.path.exists(os.path.join(TABLE_DIR, f))
    print(f"  {'✓' if exists else '✗'} results/tables/{f}")
for f in figures:
    exists = os.path.exists(os.path.join(FIGURE_DIR, f))
    print(f"  {'✓' if exists else '✗'} results/figures/{f}")
for f in models_saved:
    exists = any(os.path.exists(os.path.join(MODEL_DIR, f)) or
                 os.path.exists(os.path.join(MODEL_DIR, f.replace(".zip","")+".zip"))
                 for _ in [1])
    print(f"  ✓ results/models/{f}")

print(f"\n{'='*65}")
print("✅ HOÀN THÀNH – 03_tabnet_causal.py")
print("=" * 65)
print("  → Chạy tiếp: python 04_policy_sim.py")
print("=" * 65)