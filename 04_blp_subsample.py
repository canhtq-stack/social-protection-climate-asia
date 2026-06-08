"""
04_blp_subsample.py
===================
Best Linear Projection (BLP) + Institutional Quartile Subsample CATEs

Paper : "Social Protection as Climate Adaptation: Heterogeneous Effects
         on Income Inequality in Asia"
Journal: Economics of Disasters and Climate Change, Springer

Prerequisite: run 01_tabnet_causal.py first.

Outputs
-------
output/tables/blp_results.csv              → Table 5
output/tables/cate_by_tempshock_quartile.csv → Table 3
output/tables/subsample_rol_cate.csv       → Table 6
output/tables/aspire_sensitivity.csv
output/figures/blp_heterogeneity.png
output/figures/quartile_cate_rol.png
output/figures/figureA3_cate_vs_tempshock.png → Figure A3
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_FILE  = ROOT / "data"   / "panel_qrei_final.csv"
TABLE_DIR  = ROOT / "output" / "tables"
FIGURE_DIR = ROOT / "output" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

Y_COL  = "delta_gini"
T_COL  = "high_social_prot"
X_COLS = [
    "temp_shock", "extreme_temp_shock", "rice_yield_dev",
    "log_gdp_pc", "rule_of_law", "democracy_electoral", "disaster_count_cy",
]

# ── Helpers ────────────────────────────────────────────────────────────────

def section(title, step):
    print(f"\n{'='*70}\n[{step}] {title}\n{'='*70}")

def save_table(df, filename, label=""):
    (TABLE_DIR / filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / filename, index=True)
    print(f"  ✓ {filename}" + (f"  [{label}]" if label else ""))

def save_figure(fig, filename):
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {filename}")

def prep_arrays(df, y_col=Y_COL, t_col=T_COL, x_cols=X_COLS):
    """Return (Y, T, X, df_clean, x_avail) or None if insufficient data."""
    if t_col not in df.columns:
        df = df.copy()
        df[t_col] = 0

    x_avail = [c for c in x_cols if c in df.columns]
    id_cols = [c for c in ["country_code", "year"] if c in df.columns]
    df_sub  = df[id_cols + [y_col, t_col] + x_avail].copy()

    for col in x_avail:
        if df_sub[col].isnull().any():
            df_sub[col] = (
                df_sub.groupby("country_code")[col]
                .transform(lambda x: x.fillna(x.median()))
                if "country_code" in df_sub.columns
                else df_sub[col]
            )
            df_sub[col] = df_sub[col].fillna(df_sub[col].median()).fillna(0)

    df_sub = df_sub.dropna(subset=x_avail)
    df_sub = df_sub[df_sub[y_col].notna() & df_sub[t_col].notna()]

    if len(df_sub) < 50:
        return None

    Y = df_sub[y_col].values.astype(np.float64)
    T = (df_sub[t_col].values >= 0.5).astype(int)
    X = df_sub[x_avail].values.astype(np.float64)
    return Y, T, X, df_sub, x_avail

def fit_cf(Y, T, X, n_estimators=500):
    """Fit CausalForestDML; return (model, ate, ci_lo, ci_hi)."""
    valid = X.std(axis=0) > 1e-10
    X_fit = X[:, valid]
    if X_fit.shape[1] == 0:
        raise ValueError("No valid X columns.")

    cf = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        discrete_treatment=True,
        n_estimators=n_estimators,
        min_samples_leaf=10,
        max_samples=0.5,
        inference=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    cf.fit(Y=Y, T=T, X=X_fit, W=None)
    ate = float(cf.ate(X_fit))
    ci  = cf.ate_interval(X_fit, alpha=0.05)
    return cf, X_fit, ate, float(ci[0]), float(ci[1])

# ── Load data ──────────────────────────────────────────────────────────────
print("="*70)
print("04_blp_subsample.py — BLP + Quartile Subsample Analysis")
print("="*70)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  Loaded: {df.shape}")

# ── STEP 1: BLP — Joint regression with bootstrap SE ──────────────────────
section("BEST LINEAR PROJECTION (BLP) — Joint + Bootstrap SE", 1)

arrays = prep_arrays(df, Y_COL, T_COL, X_COLS)
if arrays is None:
    raise RuntimeError("Insufficient data for BLP estimation.")

Y_blp, T_blp, X_blp, df_clean, x_avail = arrays
print(f"  N = {len(Y_blp)}  |  Treated: {T_blp.sum()} ({T_blp.mean():.1%})")

cf_model, X_fit_blp, ate, ci_lo, ci_hi = fit_cf(Y_blp, T_blp, X_blp)
print(f"  ATE = {ate:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

# Load ATE from file if available (for consistency with 01_tabnet_causal.py)
ate_file = TABLE_DIR / "ate_summary.csv"
if ate_file.exists():
    try:
        _df = pd.read_csv(ate_file)
        ate   = float(_df["ATE"].iloc[0])
        ci_lo = float(_df["CI_lower"].iloc[0])
        ci_hi = float(_df["CI_upper"].iloc[0])
        print(f"  [ATE loaded from ate_summary.csv: {ate:.4f}]")
    except Exception:
        pass

cate_vals  = cf_model.effect(X_fit_blp)
X_demeaned = X_blp - X_blp.mean(axis=0)
X_ols      = np.column_stack([np.ones(len(cate_vals)), X_demeaned])
coef, _, _, _ = np.linalg.lstsq(X_ols, cate_vals, rcond=None)

# Save individual CATEs (required by 06_blp_cluster_bootstrap.py)
id_cols = [c for c in ["country_code", "year"] if c in df_clean.columns]
df_cate_ind = df_clean[id_cols].copy().reset_index(drop=True)
df_cate_ind["cate"] = cate_vals
for i, col in enumerate(x_avail):
    df_cate_ind[col] = X_blp[:, i]
save_table(df_cate_ind, "cate_individual.csv", "Individual CATE per obs")

# Bootstrap SE (500 replications)
N_BOOT = 500
boot_coefs = np.zeros((N_BOOT, len(x_avail)))
print(f"  Bootstrap SE ({N_BOOT} replications)...")
for b in range(N_BOOT):
    idx = np.random.choice(len(cate_vals), len(cate_vals), replace=True)
    X_b   = X_blp[idx]
    c_b   = cate_vals[idx]
    Xd_b  = X_b - X_b.mean(axis=0)
    Xo_b  = np.column_stack([np.ones(len(c_b)), Xd_b])
    cb, _, _, _ = np.linalg.lstsq(Xo_b, c_b, rcond=None)
    boot_coefs[b] = cb[1:]

blp_rows = []
for i, var in enumerate(x_avail):
    theta  = coef[i + 1]
    se     = boot_coefs[:, i].std(ddof=1)
    t_stat = theta / (se + 1e-10)
    p_val  = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    blp_rows.append({
        "Moderator":   var,
        "theta_1":     round(theta, 4),
        "SE":          round(se, 4),
        "t_stat":      round(t_stat, 3),
        "p_value":     round(p_val, 4),
        "CI_lower":    round(theta - 1.96 * se, 4),
        "CI_upper":    round(theta + 1.96 * se, 4),
        "Significant": "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "—"),
    })
    print(f"  {var:<25} θ₁ = {theta:+.4f}  SE = {se:.4f}  p = {p_val:.4f}")

df_blp = pd.DataFrame(blp_rows).set_index("Moderator")
save_table(df_blp, "blp_results.csv", "Table 5 — BLP moderation")

# BLP figure
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(df_blp))
colors_blp = ["#4472C4" if s in ("*", "**") else "#BFBFBF"
               for s in df_blp["Significant"]]
ax.barh(y_pos, df_blp["theta_1"], color=colors_blp, alpha=0.85)
ax.errorbar(df_blp["theta_1"], y_pos,
            xerr=[df_blp["theta_1"] - df_blp["CI_lower"],
                  df_blp["CI_upper"] - df_blp["theta_1"]],
            fmt="none", color="black", capsize=4)
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_blp.index, fontsize=9)
ax.set_xlabel("BLP Moderation Coefficient θ₁")
ax.set_title("Best Linear Projection: Conditional Moderation\n"
             "Blue = significant at 5%  |  500-replication bootstrap SE")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
save_figure(fig, "blp_heterogeneity.png")

# ── STEP 2: Table 3 — CATE by temperature shock quartile ─────────────────
section("CATE BY TEMPERATURE SHOCK QUARTILE — Table 3", 2)

if "temp_shock" in df_cate_ind.columns:
    df_tq = df_cate_ind[["cate", "temp_shock"]].dropna().copy()
    df_tq["quartile"] = pd.qcut(
        df_tq["temp_shock"], q=4,
        labels=["Q1(Lowest)", "Q2", "Q3", "Q4(Extreme)"],
    )
    tq_rows = []
    for q in ["Q1(Lowest)", "Q2", "Q3", "Q4(Extreme)"]:
        dq = df_tq[df_tq["quartile"] == q]
        if len(dq) < 5:
            continue
        tq_rows.append({
            "Shock_Quartile":  q,
            "CATE_mean":       round(dq["cate"].mean(), 4),
            "CI_lower":        round(np.percentile(dq["cate"], 2.5), 4),
            "CI_upper":        round(np.percentile(dq["cate"], 97.5), 4),
            "N_obs":           len(dq),
        })
    df_tq_out = pd.DataFrame(tq_rows).set_index("Shock_Quartile")
    print(df_tq_out.to_string())
    save_table(df_tq_out, "cate_by_tempshock_quartile.csv", "Table 3")

# ── STEP 3: Table 6 — Quartile CATE by Rule of Law ───────────────────────
section("INSTITUTIONAL QUARTILE CATE (Rule of Law) — Table 6", 3)

if "rule_of_law" not in df.columns:
    print("  WARNING: rule_of_law not found — skipping")
else:
    df_rol = df.copy()
    df_rol["rol_quartile"] = pd.qcut(
        df_rol["rule_of_law"], q=4,
        labels=["Q1(Weakest)", "Q2", "Q3", "Q4(Strongest)"],
    )
    q_results = []
    for q in ["Q1(Weakest)", "Q2", "Q3", "Q4(Strongest)"]:
        df_q = df_rol[df_rol["rol_quartile"] == q].copy()
        out_q = prep_arrays(df_q, Y_COL, T_COL, X_COLS)
        if out_q is None:
            print(f"  {q}: insufficient data — skipped")
            q_results.append({"Quartile": q, "ATE": np.nan, "CI_lower": np.nan,
                               "CI_upper": np.nan, "N_obs": len(df_q), "N_treated": np.nan})
            continue
        Y_q, T_q, X_q, _, _ = out_q
        if T_q.sum() == 0 or T_q.sum() == len(T_q):
            print(f"  {q}: no treatment variation — skipped (Treated={T_q.sum()}/{len(T_q)})")
            q_results.append({"Quartile": q, "ATE": np.nan, "CI_lower": np.nan,
                               "CI_upper": np.nan, "N_obs": len(T_q), "N_treated": int(T_q.sum())})
            continue
        try:
            _, _, ate_q, ci_lo_q, ci_hi_q = fit_cf(Y_q, T_q, X_q)
            print(f"  {q}: ATE={ate_q:.4f}  [{ci_lo_q:.4f}, {ci_hi_q:.4f}]  N={len(T_q)}")
            q_results.append({"Quartile": q, "ATE": round(ate_q, 4),
                               "CI_lower": round(ci_lo_q, 4), "CI_upper": round(ci_hi_q, 4),
                               "N_obs": len(T_q), "N_treated": int(T_q.sum())})
        except Exception as e:
            print(f"  {q} ERROR: {e}")
            q_results.append({"Quartile": q, "ATE": np.nan, "CI_lower": np.nan,
                               "CI_upper": np.nan, "N_obs": len(T_q), "N_treated": int(T_q.sum())})

    df_quartile = pd.DataFrame(q_results).set_index("Quartile")
    print(df_quartile.to_string())
    save_table(df_quartile, "subsample_rol_cate.csv", "Table 6 — Quartile CATE by RoL")

    # Figure
    valid = df_quartile["ATE"].notna()
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ["#4472C4" if a < 0 else "#ED7D31"
                  for a in df_quartile.loc[valid, "ATE"]]
    ax.bar(df_quartile.index[valid], df_quartile.loc[valid, "ATE"],
           color=bar_colors, alpha=0.8, edgecolor="white")
    ax.errorbar(df_quartile.index[valid], df_quartile.loc[valid, "ATE"],
                yerr=[df_quartile.loc[valid, "ATE"] - df_quartile.loc[valid, "CI_lower"],
                      df_quartile.loc[valid, "CI_upper"] - df_quartile.loc[valid, "ATE"]],
                fmt="none", color="black", capsize=5)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Institutional Quality Quartile (Rule of Law)")
    ax.set_ylabel("ATE (Effect on ΔGini)")
    ax.set_title("Heterogeneous ATE by Institutional Quality Quartile\n"
                 "Negative = social protection reduces inequality")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "quartile_cate_rol.png")

# ── STEP 4: Figure A3 — CATEs vs temperature shock intensity ──────────────
section("CATE vs TEMPERATURE SHOCK INTENSITY — Figure A3", 4)

if "temp_shock" in df_cate_ind.columns:
    df_a3 = df_cate_ind[["cate", "temp_shock", "country_code"]].dropna()
    cate_norm = (
        (df_a3["cate"] - df_a3["cate"].min())
        / (df_a3["cate"].max() - df_a3["cate"].min() + 1e-10)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df_a3["temp_shock"], df_a3["cate"],
                    c=cate_norm, cmap="RdYlBu_r", alpha=0.55, s=25)
    plt.colorbar(sc, ax=ax, label="CATE (blue = most negative → red = most positive)")
    ax.axhline(ate, color="#E07B39", linewidth=2, label=f"Full-sample ATE = {ate:.3f}")
    ax.axhline(0,   color="black",   linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline( 2,  color="#888888", linewidth=1, linestyle=":", alpha=0.7, label="|z| = 2")
    ax.axvline(-2,  color="#888888", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_xlabel("Temperature Shock Intensity (z-score, detrended)")
    ax.set_ylabel("CATE — Effect on ΔGini")
    ax.set_title(
        f"Figure A3. Observation-Level CATEs vs Temperature Shock Intensity\n"
        f"N = {len(df_a3)} country-year observations  |  CausalForestDML"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, linestyle=":")
    plt.tight_layout()
    save_figure(fig, "figureA3_cate_vs_tempshock.png")

print(f"\nDONE — 04_blp_subsample.py")
print("  Next step: python code/05_parametric_benchmark.py")
