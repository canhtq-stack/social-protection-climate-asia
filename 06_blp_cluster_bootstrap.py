"""
06_blp_cluster_bootstrap.py
===========================
BLP with Cluster-Robust Standard Errors (country-level clustering)
Cross-check for Table 5 (complements bootstrap SE from 04_blp_subsample.py)

Paper : "Social Protection as Climate Adaptation: Heterogeneous Effects
         on Income Inequality in Asia"
Journal: Economics of Disasters and Climate Change, Springer

Prerequisites: run 01_tabnet_causal.py AND 04_blp_subsample.py first.

Outputs
-------
output/tables/blp_clustered_se.csv   → robustness check for Table 5
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
CATE_FILE  = ROOT / "output" / "tables" / "cate_individual.csv"
DATA_FILE  = ROOT / "data"   / "panel_qrei_final.csv"
TABLE_DIR  = ROOT / "output" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

MODERATORS = [
    "rule_of_law", "democracy_electoral", "log_gdp_pc",
    "temp_shock", "extreme_temp_shock", "rice_yield_dev", "disaster_count_cy",
]

# ── Load data ──────────────────────────────────────────────────────────────
print("="*70)
print("06_blp_cluster_bootstrap.py — BLP with Clustered SE")
print("="*70)

if not CATE_FILE.exists():
    raise FileNotFoundError(
        f"CATE file not found: {CATE_FILE}\n"
        "→ Run 01_tabnet_causal.py and 04_blp_subsample.py first."
    )
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

df_cate  = pd.read_csv(CATE_FILE,  low_memory=False)
df_panel = pd.read_csv(DATA_FILE,  low_memory=False)
print(f"  CATE file  : {df_cate.shape}")
print(f"  Panel file : {df_panel.shape}")

# Merge panel moderators not already in CATE file
mods_in_cate  = [m for m in MODERATORS if m in df_cate.columns]
mods_to_merge = [m for m in MODERATORS if m in df_panel.columns and m not in mods_in_cate]

if mods_to_merge:
    merge_cols   = ["country_code", "year"] + mods_to_merge
    df_panel_sub = df_panel[merge_cols].drop_duplicates(subset=["country_code", "year"])
    df = df_cate.merge(df_panel_sub, on=["country_code", "year"], how="left")
else:
    df = df_cate.copy()

available_mods = [m for m in MODERATORS if m in df.columns]
print(f"  Available moderators: {available_mods}")
print(f"  CATE non-NaN: {df['cate'].notna().sum() if 'cate' in df.columns else 'missing'}")

# ── BLP with clustered SE ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print("BLP REGRESSION — Cluster-Robust SE (country_code)")
print("="*65)

results = []

for mod in available_mods:
    sub = df[["cate", mod, "country_code"]].dropna()
    n_obs     = len(sub)
    n_clusters = sub["country_code"].nunique()

    if n_obs < 30:
        print(f"  {mod:<25}: skipped (N={n_obs} < 30)")
        continue

    mod_demeaned = sub[mod] - sub[mod].mean()
    X = sm.add_constant(mod_demeaned)
    y = sub["cate"]

    try:
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["country_code"]},
        )
        theta1 = model.params.iloc[1]
        se     = model.bse.iloc[1]
        t      = model.tvalues.iloc[1]
        p      = model.pvalues.iloc[1]
        ci     = model.conf_int().iloc[1]
        sig    = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "—"))

        results.append({
            "Moderator":    mod,
            "theta1":       round(theta1, 4),
            "SE_clustered": round(se, 4),
            "t_clustered":  round(t, 3),
            "p_clustered":  round(p, 4),
            "CI_lower":     round(ci.iloc[0], 4),
            "CI_upper":     round(ci.iloc[1], 4),
            "Significant":  sig,
            "N_obs":        n_obs,
            "N_clusters":   n_clusters,
        })
        print(f"  {mod:<25}  θ₁ = {theta1:+.4f}  SE = {se:.4f}  p = {p:.4f}  {sig}")

    except Exception as e:
        print(f"  {mod:<25}: ERROR — {e}")

# ── Save ───────────────────────────────────────────────────────────────────

df_results = pd.DataFrame(results)

if df_results.empty:
    print("\nWARNING: No moderators produced valid estimates.")
    print("  Diagnostics:")
    for mod in available_mods:
        sub = df[["cate", mod, "country_code"]].dropna()
        print(f"    {mod:<25}  N = {len(sub)},  clusters = {sub['country_code'].nunique()}")
else:
    df_results.to_csv(TABLE_DIR / "blp_clustered_se.csv", index=False)
    print(f"\n  ✓ blp_clustered_se.csv  [cluster-robust cross-check for Table 5]")
    print(f"  {len(df_results)} moderator(s) estimated.\n")
    print(df_results[["Moderator", "theta1", "SE_clustered", "p_clustered", "Significant"]]
          .to_string(index=False))

print(f"\nDONE — 06_blp_cluster_bootstrap.py")
print("  Next step: python code/07_calc_agric_median.py")
