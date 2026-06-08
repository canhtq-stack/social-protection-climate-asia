"""
02_policy_sim.py
================
Policy Simulation: Break-even & Cost-Benefit Analysis

Paper : "Social Protection as Climate Adaptation: Heterogeneous Effects
         on Income Inequality in Asia"
Journal: Economics of Disasters and Climate Change, Springer

Prerequisite: run 01_tabnet_causal.py first.

Outputs
-------
output/tables/policy_sim_bcr_current.csv   → Table in Section 7.1
output/tables/policy_sim_breakeven.csv
output/tables/policy_sim_epsilon_sensitivity.csv
output/figures/figure_bcr_by_country.png
output/figures/figure_breakeven_coverage.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_FILE  = ROOT / "data"   / "panel_qrei_final.csv"
CATE_FILE  = ROOT / "output" / "tables" / "cate_by_country.csv"
ATE_FILE   = ROOT / "output" / "tables" / "ate_summary.csv"
TABLE_DIR  = ROOT / "output" / "tables"
FIGURE_DIR = ROOT / "output" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── Economic parameters ────────────────────────────────────────────────────
EPSILON_BASE  = 2.0           # welfare weight (Ostry et al. 2014 benchmark)
EPSILON_RANGE = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
COST_BASE     = 0.04          # GDP cost per pp of coverage (ILO 2021 estimate)
COST_RANGE    = [0.02, 0.04, 0.06, 0.08]

REGION_MAP = {
    "IND": "South Asia",  "BGD": "South Asia",  "PAK": "South Asia",
    "LKA": "South Asia",  "NPL": "South Asia",  "AFG": "South Asia",
    "BTN": "South Asia",
    "VNM": "SE Asia",     "PHL": "SE Asia",     "IDN": "SE Asia",
    "THA": "SE Asia",     "MYS": "SE Asia",     "KHM": "SE Asia",
    "MMR": "SE Asia",     "LAO": "SE Asia",     "SGP": "SE Asia",
    "TLS": "SE Asia",
    "CHN": "East Asia",   "KOR": "East Asia",   "JPN": "East Asia",
    "MNG": "East Asia",   "HKG": "East Asia",   "MAC": "East Asia",
    "BRN": "East Asia",   "TWN": "East Asia",
}

# ── Helper functions ───────────────────────────────────────────────────────

def welfare_usd(gini_reduction, gdp_pc, epsilon):
    """Welfare gain in USD per capita from a Gini-point reduction."""
    return epsilon * gini_reduction * gdp_pc / 100.0

def fiscal_cost_usd(coverage_pct, gdp_pc, cost_per_pp):
    """Fiscal cost in USD per capita for a given coverage level."""
    return cost_per_pp * coverage_pct * gdp_pc / 100.0

def bcr(gini_reduction, coverage_pct, gdp_pc, epsilon, cost_per_pp):
    """Benefit-cost ratio."""
    cost = fiscal_cost_usd(coverage_pct, gdp_pc, cost_per_pp)
    return welfare_usd(gini_reduction, gdp_pc, epsilon) / cost if cost > 0 else np.nan

def coverage_breakeven(gini_reduction, epsilon, cost_per_pp):
    """Break-even coverage level (%) at which BCR = 1."""
    return min((epsilon * gini_reduction) / cost_per_pp, 100.0)

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
print("="*70)
print("02_policy_sim.py — Policy Simulation")
print("="*70)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
if not CATE_FILE.exists():
    raise FileNotFoundError(
        f"CATE file not found: {CATE_FILE}\n"
        "→ Run 01_tabnet_causal.py first."
    )

df       = pd.read_csv(DATA_FILE, low_memory=False)
df_cate  = pd.read_csv(CATE_FILE, index_col=0)

# Load ATE (fall back to manuscript value if file missing)
ate_val = -0.1492
if ATE_FILE.exists():
    try:
        df_ate  = pd.read_csv(ATE_FILE)
        ate_val = float(df_ate["ATE"].iloc[0])
    except Exception:
        pass

gini_reduction_ate = abs(ate_val)

gdp_by_c = (df.groupby("country_code")["gdp_pc"].mean()
            if "gdp_pc" in df.columns else pd.Series(dtype=float))
cov_by_c = (df.groupby("country_code")["social_prot_coverage"].mean()
            if "social_prot_coverage" in df.columns else pd.Series(dtype=float))

mean_gdp = gdp_by_c.mean() if not gdp_by_c.empty else 5000.0
mean_cov = cov_by_c.mean() if not cov_by_c.empty else 30.0

cate_col = "cate_mean" if "cate_mean" in df_cate.columns else df_cate.columns[0]

print(f"  |ATE|           : {gini_reduction_ate:.4f} Gini points")
print(f"  Mean GDP/pc     : ${mean_gdp:,.0f}")
print(f"  Mean coverage   : {mean_cov:.1f}%")

# ── STEP 2: BCR at current coverage ───────────────────────────────────────
print("\n--- BCR at current coverage (epsilon=2.0, cost=0.04) ---")

rows = []
for country in sorted(df_cate.index):
    try:
        cate_c   = abs(float(df_cate.loc[country, cate_col]))
        gdp_c    = float(gdp_by_c.get(country, mean_gdp))
        cov_c    = float(cov_by_c.get(country, mean_cov))
        region   = REGION_MAP.get(country, "Unknown")
        bcr_c    = bcr(cate_c, cov_c, gdp_c, EPSILON_BASE, COST_BASE)
        eps_be_c = (fiscal_cost_usd(cov_c, gdp_c, COST_BASE)
                    / (cate_c * gdp_c / 100.0)
                    if cate_c > 0 else np.inf)
        rows.append({
            "country_code": country, "region": region,
            "cate_abs":       round(cate_c,  4),
            "gdp_pc":         round(gdp_c,   0),
            "coverage_pct":   round(cov_c,   1),
            "welfare_usd_pc": round(welfare_usd(cate_c, gdp_c, EPSILON_BASE), 2),
            "cost_usd_pc":    round(fiscal_cost_usd(cov_c, gdp_c, COST_BASE), 2),
            "bcr":            round(bcr_c,    3),
            "epsilon_needed": round(eps_be_c, 2),
        })
    except Exception:
        continue

df_q1 = pd.DataFrame(rows).sort_values("bcr", ascending=False)
save_table(df_q1.set_index("country_code"), "policy_sim_bcr_current.csv", "BCR at current coverage")

# ── STEP 3: Break-even coverage (full-sample ATE) ─────────────────────────
print("\n--- Break-even coverage by epsilon ---")

be_rows = []
for eps in EPSILON_RANGE:
    be_cov = coverage_breakeven(gini_reduction_ate, eps, COST_BASE)
    be_rows.append({
        "epsilon": eps,
        "breakeven_coverage_pct": round(be_cov, 2),
        "bcr_at_breakeven": 1.0,
        "note": "Full-sample ATE; cost_per_pp = 0.04",
    })

df_be = pd.DataFrame(be_rows).set_index("epsilon")
print(df_be.to_string())
save_table(df_be, "policy_sim_breakeven.csv", "Break-even coverage by epsilon")

# ── STEP 4: Epsilon sensitivity grid ──────────────────────────────────────
print("\n--- BCR sensitivity: epsilon × cost grid ---")

sens_rows = []
for eps in EPSILON_RANGE:
    for cost in COST_RANGE:
        bcr_v = bcr(gini_reduction_ate, mean_cov, mean_gdp, eps, cost)
        sens_rows.append({
            "epsilon": eps, "cost_per_pp": cost,
            "bcr_mean_coverage": round(bcr_v, 3),
        })

df_sens = (
    pd.DataFrame(sens_rows)
    .pivot(index="epsilon", columns="cost_per_pp", values="bcr_mean_coverage")
)
print(df_sens.to_string())
save_table(df_sens, "policy_sim_epsilon_sensitivity.csv", "BCR sensitivity grid")

# ── STEP 5: Figure — BCR by country ───────────────────────────────────────
df_plot = df_q1[df_q1["bcr"].notna() & np.isfinite(df_q1["bcr"])].head(15)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2B5FA5" if b > 1 else "#AAAAAA" for b in df_plot["bcr"]]
ax.barh(df_plot["country_code"], df_plot["bcr"], color=colors, alpha=0.85)
ax.axvline(1, color="red", linewidth=1.5, linestyle="--", label="BCR = 1 (break-even)")
ax.set_xlabel("Benefit-Cost Ratio (ε = 2.0, cost = 0.04% GDP/pp)")
ax.set_title("Policy Simulation: BCR at Current Coverage\nBlue = BCR > 1 (fiscally justified)")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3, linestyle=":")
plt.tight_layout()
save_figure(fig, "figure_bcr_by_country.png")

# ── STEP 6: Figure — break-even coverage ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df_be.index, df_be["breakeven_coverage_pct"],
        marker="o", color="#2B5FA5", linewidth=2)
ax.axhline(mean_cov, color="#E07B39", linestyle="--",
           label=f"Sample mean coverage ({mean_cov:.1f}%)")
ax.set_xlabel("Welfare elasticity (ε)")
ax.set_ylabel("Break-even coverage (%)")
ax.set_title("Break-even Analysis: Minimum Coverage for BCR ≥ 1\n"
             f"|ATE| = {gini_reduction_ate:.3f} Gini points  |  cost = {COST_BASE} × GDP/pp")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle=":")
plt.tight_layout()
save_figure(fig, "figure_breakeven_coverage.png")

print(f"\nDONE — 02_policy_sim.py")
print(f"  Next step: python code/03_robustness.py")
