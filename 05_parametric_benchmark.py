"""
05_parametric_benchmark.py
==========================
Parametric OLS Benchmark with Country/Year Fixed Effects and Interaction Terms
(Table 8 in manuscript)

Paper : "Social Protection as Climate Adaptation: Heterogeneous Effects
         on Income Inequality in Asia"
Journal: Economics of Disasters and Climate Change, Springer

Outputs
-------
output/tables/parametric_benchmark.csv  → Table 8
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data"   / "panel_qrei_final.csv"
TABLE_DIR = ROOT / "output" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Variable definitions ───────────────────────────────────────────────────
Y_COL          = "delta_gini"
T_COL          = "high_social_prot"
SHOCK_COL      = "temp_shock"
ROL_COL        = "rule_of_law"
CONTROLS       = ["log_gdp_pc", "democracy_electoral", "gini_lag1", "temp_shock_lag1"]

# ── Load data ──────────────────────────────────────────────────────────────
print("="*70)
print("05_parametric_benchmark.py — OLS Benchmark (Table 8)")
print("="*70)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  Loaded: {df.shape}")

df = df.copy()
df["country_code"] = df["country_code"].astype(str)
df["year"]         = df["year"].astype(str)

# Create interaction terms
df["sp_x_temp_shock"]  = (df[T_COL] * df[SHOCK_COL]
                          if SHOCK_COL in df.columns
                          else pd.Series(0.0, index=df.index))
df["sp_x_rule_of_law"] = (df[T_COL] * df[ROL_COL]
                          if ROL_COL in df.columns
                          else pd.Series(0.0, index=df.index))

# Drop rows with missing required variables
req_cols = [Y_COL, T_COL] + [c for c in [SHOCK_COL, ROL_COL] + CONTROLS if c in df.columns]
df_model = df.dropna(subset=req_cols).copy()
print(f"  Estimation sample: {len(df_model):,} obs  |  "
      f"{df_model['country_code'].nunique()} countries")

controls_str = " + ".join([c for c in CONTROLS if c in df_model.columns])
has_rol      = ROL_COL in df_model.columns and df_model[ROL_COL].std() > 0
fe_str       = "C(country_code) + C(year)"

# ── OLS estimation ─────────────────────────────────────────────────────────

def fit_ols(formula, name):
    print(f"  Estimating {name}...")
    try:
        m = smf.ols(formula, data=df_model).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_model["country_code"]},
        )
        print(f"    N = {int(m.nobs)},  R² = {m.rsquared:.4f},  R²_adj = {m.rsquared_adj:.4f}")
        return m
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

base = f"{Y_COL} ~ {T_COL} + {SHOCK_COL}"
if has_rol:
    base += f" + {ROL_COL}"
base += f" + {controls_str}" if controls_str else ""
base += f" + {fe_str}"

m1 = fit_ols(base, "Model 1 — Baseline")

f2  = f"{Y_COL} ~ {T_COL} + {SHOCK_COL} + sp_x_temp_shock"
if has_rol:
    f2 += f" + {ROL_COL}"
f2 += f" + {controls_str} + {fe_str}" if controls_str else f" + {fe_str}"
m2 = fit_ols(f2, "Model 2 — + T×Shock")

f3 = f"{Y_COL} ~ {T_COL} + {SHOCK_COL}"
if has_rol:
    f3 += f" + {ROL_COL} + sp_x_rule_of_law"
f3 += f" + {controls_str} + {fe_str}" if controls_str else f" + {fe_str}"
m3 = fit_ols(f3, "Model 3 — + T×RoL")

f4 = f"{Y_COL} ~ {T_COL} + {SHOCK_COL} + sp_x_temp_shock"
if has_rol:
    f4 += f" + {ROL_COL} + sp_x_rule_of_law"
f4 += f" + {controls_str} + {fe_str}" if controls_str else f" + {fe_str}"
m4 = fit_ols(f4, "Model 4 — Full")

# ── Build results table ────────────────────────────────────────────────────

KEY_VARS = [T_COL, SHOCK_COL, "sp_x_temp_shock", ROL_COL, "sp_x_rule_of_law"] + CONTROLS

rows = {}
for var in KEY_VARS:
    row = []
    for m in [m1, m2, m3, m4]:
        if m is None or var not in m.params:
            row.append("—")
        else:
            b  = m.params[var]
            se = m.bse[var]
            p  = m.pvalues[var]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            row.append(f"{b:.3f}{stars} ({se:.3f})")
    rows[var] = row

# Add model-level stats
for stat_name, attr in [("N", "nobs"), ("R²", "rsquared"), ("R²_adj", "rsquared_adj")]:
    rows[stat_name] = [
        (str(int(m.nobs)) if attr == "nobs" else f"{getattr(m, attr):.3f}")
        if m is not None else "—"
        for m in [m1, m2, m3, m4]
    ]
rows["Country FE"] = ["Yes"] * 4
rows["Year FE"]    = ["Yes"] * 4
rows["Clustered SE"] = ["Country"] * 4

df_table = pd.DataFrame(rows, index=["Model 1", "Model 2", "Model 3", "Model 4"]).T
df_table.to_csv(TABLE_DIR / "parametric_benchmark.csv")
print("\n  ✓ parametric_benchmark.csv  [Table 8]")

# Print key interaction coefficients
if m4 is not None:
    for term in ["sp_x_temp_shock", "sp_x_rule_of_law"]:
        if term in m4.params:
            print(f"\n  Model 4 — {term}: "
                  f"β = {m4.params[term]:.4f}  "
                  f"SE = {m4.bse[term]:.4f}  "
                  f"p = {m4.pvalues[term]:.3f}")

print(f"\nDONE — 05_parametric_benchmark.py")
print("  Next step: python code/06_blp_cluster_bootstrap.py")
