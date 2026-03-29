"""
QREI Project – 06_parametric_benchmark.py
==========================================
Parametric Benchmark: Panel OLS with Interaction Terms

Mục đích:
  Chạy mô hình OLS tuyến tính với interaction terms để SO SÁNH
  với CausalForestDML (02_tabnet_causal.py).

  Kết quả kỳ vọng: interaction coefficients β₃ (high_social_prot × temp_shock)
  và β₅ (high_social_prot × rule_of_law) sẽ KHÔNG significant hoặc rất nhỏ.
  Đây là bằng chứng rằng parametric OLS "miss" heterogeneity mà
  CausalForestDML phát hiện được — justifying estimator choice của bài.

Mô hình:
  ΔGini_it = α
           + β₁·high_social_prot_it
           + β₂·temp_shock_it
           + β₃·(high_social_prot × temp_shock)_it
           + β₄·rule_of_law_it
           + β₅·(high_social_prot × rule_of_law)_it
           + β₆·log_gdp_pc_it
           + β₇·democracy_electoral_it
           + β₈·gini_lag1_it
           + β₉·temp_shock_lag1_it
           + country FE + year FE
           + ε_it

  Standard errors: clustered at country level

Paper  : "Social Protection as Climate Adaptation:
          Heterogeneous Effects on Income Inequality in Asia"
Journal: Environmental and Resource Economics (ERE) – Springer

INPUT  : Data/processed/panel_qrei_final.csv
OUTPUT : results/tables/parametric_benchmark.csv   ← bảng chính cho paper
         results/tables/parametric_benchmark_full.csv  ← full output

Chạy   : python 06_parametric_benchmark.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH – chỉnh BASE cho phù hợp với máy bạn
# ============================================================

BASE = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\Quantum_Global Economic Resilience\qrei_project"

DATA_FILE  = os.path.join(BASE, "Data", "processed", "panel_qrei_final.csv")
TABLE_DIR  = os.path.join(BASE, "results", "tables")

os.makedirs(TABLE_DIR, exist_ok=True)

# ============================================================
# VARIABLE NAMES – khớp với 02_tabnet_causal.py
# ============================================================

Y_COL = "delta_gini"
T_COL = "high_social_prot"

# Controls giống W_COLS trong CausalForestDML
CONTROLS = [
    "log_gdp_pc",
    "democracy_electoral",
    "gini_lag1",
    "temp_shock_lag1",
]

# Interaction variables
INTERACT_T_SHOCK = "temp_shock"
INTERACT_T_ROL   = "rule_of_law"

ID_COLS = ["country_code", "year"]

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("06_parametric_benchmark.py")
print("Parametric OLS Benchmark with Interaction Terms")
print("=" * 65)

print(f"\n[1] Loading data from:\n    {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
print(f"    Raw shape: {df.shape}")

# ============================================================
# PREPARE VARIABLES
# ============================================================

print("\n[2] Preparing variables...")

# Interaction terms
df["sp_x_temp_shock"]  = df[T_COL] * df[INTERACT_T_SHOCK]
df["sp_x_rule_of_law"] = df[T_COL] * df[INTERACT_T_ROL]

# Drop rows with missing outcome or treatment
required_cols = [Y_COL, T_COL, INTERACT_T_SHOCK, INTERACT_T_ROL] + CONTROLS
df_model = df.dropna(subset=required_cols).copy()

# ------------------------------------------------------------------
# FIX LỖI StringDtype: Ép kiểu về numpy native types (object/float64)
# ------------------------------------------------------------------
# 1. Biến cố định (FE) phải là object để statsmodels nhận diện đúng khi dùng C()
df_model["country_code"] = df_model["country_code"].astype(object)
df_model["year"]         = df_model["year"].astype(object)

# 2. Duyệt toàn bộ cột để xử lý các dtype mở rộng của pandas (Int64, string, etc.)
for _col in df_model.columns:
    # Kiểm tra xem có phải là pandas extension dtype không (nguyên nhân gây lỗi)
    if pd.api.types.is_extension_array_dtype(df_model[_col]):
        # Nếu là số -> chuyển về float64
        if pd.api.types.is_numeric_dtype(df_model[_col]):
            df_model[_col] = df_model[_col].astype(np.float64)
        # Nếu là chuỗi/boolean -> chuyển về object
        else:
            df_model[_col] = df_model[_col].astype(object)
    # Đảm bảo các cột số bình thường cũng là float64 để tránh warning
    elif pd.api.types.is_numeric_dtype(df_model[_col]):
        df_model[_col] = df_model[_col].astype(np.float64)

print(f"    Estimation sample N = {len(df_model)}")
print(f"    Countries: {df_model['country_code'].nunique()}")
print(f"    Years    : {df_model['year'].nunique()}")
print(f"    Treated obs (high_social_prot=1): {df_model[T_COL].sum():.0f} "
      f"({df_model[T_COL].mean()*100:.1f}%)")

# ============================================================
# MODEL 1: Baseline — no interactions
# ============================================================

print("\n[3] Estimating Model 1 (baseline, no interactions)...")

formula_base = (
    f"{Y_COL} ~ {T_COL} + {INTERACT_T_SHOCK} + {INTERACT_T_ROL} "
    f"+ {' + '.join(CONTROLS)} "
    f"+ C(country_code) + C(year)"
)

m1 = smf.ols(formula_base, data=df_model).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_model["country_code"]}
)

print(f"    N = {int(m1.nobs)}, R² = {m1.rsquared:.4f}, "
      f"R²_adj = {m1.rsquared_adj:.4f}")
print(f"    high_social_prot: β = {m1.params[T_COL]:.4f}, "
      f"p = {m1.pvalues[T_COL]:.3f}")

# ============================================================
# MODEL 2: Add interaction with temp_shock (H1)
# ============================================================

print("\n[4] Estimating Model 2 (+ high_social_prot × temp_shock)...")

formula_m2 = (
    f"{Y_COL} ~ {T_COL} + {INTERACT_T_SHOCK} + sp_x_temp_shock "
    f"+ {INTERACT_T_ROL} "
    f"+ {' + '.join(CONTROLS)} "
    f"+ C(country_code) + C(year)"
)

m2 = smf.ols(formula_m2, data=df_model).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_model["country_code"]}
)

print(f"    N = {int(m2.nobs)}, R² = {m2.rsquared:.4f}")
print(f"    high_social_prot:           β = {m2.params[T_COL]:.4f}, "
      f"p = {m2.pvalues[T_COL]:.3f}")
print(f"    high_social_prot × temp:    β = {m2.params['sp_x_temp_shock']:.4f}, "
      f"p = {m2.pvalues['sp_x_temp_shock']:.3f}")

# ============================================================
# MODEL 3: Add interaction with rule_of_law (H2)
# ============================================================

print("\n[5] Estimating Model 3 (+ high_social_prot × rule_of_law)...")

formula_m3 = (
    f"{Y_COL} ~ {T_COL} + {INTERACT_T_SHOCK} + {INTERACT_T_ROL} "
    f"+ sp_x_rule_of_law "
    f"+ {' + '.join(CONTROLS)} "
    f"+ C(country_code) + C(year)"
)

m3 = smf.ols(formula_m3, data=df_model).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_model["country_code"]}
)

print(f"    N = {int(m3.nobs)}, R² = {m3.rsquared:.4f}")
print(f"    high_social_prot:            β = {m3.params[T_COL]:.4f}, "
      f"p = {m3.pvalues[T_COL]:.3f}")
print(f"    high_social_prot × rol:      β = {m3.params['sp_x_rule_of_law']:.4f}, "
      f"p = {m3.pvalues['sp_x_rule_of_law']:.3f}")

# ============================================================
# MODEL 4: Full model — both interactions (primary benchmark)
# ============================================================

print("\n[6] Estimating Model 4 (full: both interactions)...")

formula_m4 = (
    f"{Y_COL} ~ {T_COL} "
    f"+ {INTERACT_T_SHOCK} + sp_x_temp_shock "
    f"+ {INTERACT_T_ROL} + sp_x_rule_of_law "
    f"+ {' + '.join(CONTROLS)} "
    f"+ C(country_code) + C(year)"
)

m4 = smf.ols(formula_m4, data=df_model).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_model["country_code"]}
)

print(f"    N = {int(m4.nobs)}, R² = {m4.rsquared:.4f}")
print(f"    high_social_prot:            β = {m4.params[T_COL]:.4f}, "
      f"p = {m4.pvalues[T_COL]:.3f}")
print(f"    high_social_prot × temp:     β = {m4.params['sp_x_temp_shock']:.4f}, "
      f"p = {m4.pvalues['sp_x_temp_shock']:.3f}")
print(f"    high_social_prot × rol:      β = {m4.params['sp_x_rule_of_law']:.4f}, "
      f"p = {m4.pvalues['sp_x_rule_of_law']:.3f}")

# ============================================================
# BUILD CLEAN OUTPUT TABLE (for paper Appendix)
# ============================================================

print("\n[7] Building output table...")

# Variables to extract (in display order)
key_vars = [
    T_COL,
    INTERACT_T_SHOCK,
    "sp_x_temp_shock",
    INTERACT_T_ROL,
    "sp_x_rule_of_law",
    "log_gdp_pc",
    "democracy_electoral",
    "gini_lag1",
    "temp_shock_lag1",
]

# Labels for display
labels = {
    T_COL:               "High social protection",
    INTERACT_T_SHOCK:    "Temperature shock",
    "sp_x_temp_shock":   "High soc. prot. × Temp. shock",
    INTERACT_T_ROL:      "Rule of law",
    "sp_x_rule_of_law":  "High soc. prot. × Rule of law",
    "log_gdp_pc":        "Log GDP per capita",
    "democracy_electoral": "Electoral democracy",
    "gini_lag1":         "Lagged Gini",
    "temp_shock_lag1":   "Lagged temperature shock",
}

def sig_stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

def fmt_coef(model, var):
    """Return 'β (SE)' string with significance stars, or '—' if var absent."""
    if var not in model.params:
        return "—"
    b  = model.params[var]
    se = model.bse[var]
    p  = model.pvalues[var]
    return f"{b:.3f}{sig_stars(p)} ({se:.3f})"

rows = []
for v in key_vars:
    label = labels.get(v, v)
    rows.append({
        "Variable":   label,
        "Model 1\n(Baseline)":     fmt_coef(m1, v),
        "Model 2\n(+ T×Shock)":    fmt_coef(m2, v),
        "Model 3\n(+ T×RoL)":      fmt_coef(m3, v),
        "Model 4\n(Full)":         fmt_coef(m4, v),
    })

# Add fit statistics
for stat_label, vals in [
    ("N",       [int(m.nobs) for m in [m1, m2, m3, m4]]),
    ("R²",      [f"{m.rsquared:.3f}" for m in [m1, m2, m3, m4]]),
    ("R²_adj",  [f"{m.rsquared_adj:.3f}" for m in [m1, m2, m3, m4]]),
    ("Country FE", ["Yes"] * 4),
    ("Year FE",    ["Yes"] * 4),
    ("Clustered SE", ["Country"] * 4),
]:
    rows.append({
        "Variable": stat_label,
        "Model 1\n(Baseline)":  str(vals[0]),
        "Model 2\n(+ T×Shock)": str(vals[1]),
        "Model 3\n(+ T×RoL)":   str(vals[2]),
        "Model 4\n(Full)":      str(vals[3]),
    })

df_out = pd.DataFrame(rows)

# Save
out_path = os.path.join(TABLE_DIR, "parametric_benchmark.csv")
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"    Saved: {out_path}")

# Also save a plain-text version for quick inspection
print("\n" + "=" * 65)
print("CLEAN TABLE (copy-paste này vào báo cáo):")
print("=" * 65)
print(df_out.to_string(index=False))

# ============================================================
# KEY RESULTS SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("KEY RESULTS – đọc trước khi viết vào paper:")
print("=" * 65)

b3 = m4.params.get("sp_x_temp_shock", np.nan)
p3 = m4.pvalues.get("sp_x_temp_shock", np.nan)
b5 = m4.params.get("sp_x_rule_of_law", np.nan)
p5 = m4.pvalues.get("sp_x_rule_of_law", np.nan)

print(f"""
  Model 4 (Full benchmark):
  ─────────────────────────────────────────────────────
  high_social_prot × temp_shock  (β₃): {b3:.4f}  p = {p3:.3f}  {sig_stars(p3) or "n.s."}
  high_social_prot × rule_of_law (β₅): {b5:.4f}  p = {p5:.3f}  {sig_stars(p5) or "n.s."}
  ─────────────────────────────────────────────────────

  DIỄN GIẢI:
  {'→ Cả β₃ và β₅ KHÔNG significant (p > 0.10)' if p3 > 0.10 and p5 > 0.10 else
   '→ Một hoặc cả hai interaction significant — xem ghi chú bên dưới'}

  Kỳ vọng của bài:
  β₃ (high_social_prot × temp_shock)  → không significant
     vì OLS giả định linear interaction, trong khi causal forest
     phát hiện threshold pattern (Q1–Q2 significant, Q3 gần zero).

  β₅ (high_social_prot × rule_of_law) → không significant
     vì OLS average heterogeneity thành một coefficient duy nhất,
     trong khi quartile subsample analysis cho thấy Q1 ATE = -0.706
     nhưng Q3 ATE = +0.067 — không thể capture bằng linear term.

  → Đây là bằng chứng trực tiếp justifying CausalForestDML.
    Dùng kết quả này trong Section 6.3 và Appendix.
""")

# ============================================================
# GHI CHÚ NẾU KẾT QUẢ KHÁC KỲ VỌNG
# ============================================================

if p3 < 0.10 or p5 < 0.10:
    print("  ⚠ GHI CHÚ: Một hoặc cả hai interaction SIGNIFICANT.")
    print("    Điều này không làm yếu bài — mà cần diễn giải khác:")
    print()
    if p3 < 0.10:
        print(f"    β₃ (×temp_shock) = {b3:.4f} (p={p3:.3f}) — significant")
        print("    → Parametric OLS cũng phát hiện climate-moderation.")
        print("    → Nhưng CausalForestDML cho CATE granular hơn (country-level)")
        print("      và không giả định linearity. Đây là COMPLEMENTARY evidence.")
    if p5 < 0.10:
        print(f"    β₅ (×rule_of_law) = {b5:.4f} (p={p5:.3f}) — significant")
        print("    → OLS phát hiện institutional moderation nhưng chỉ ở mức average.")
        print("    → BLP (θ₁ = 0.606) và quartile subsample (Q1: -0.706, Q3: +0.067)")
        print("      cho thấy nonlinear threshold mà linear term không capture được.")
    print()
    print("    → Khi viết vào Section 6.3, frame là:")
    print("      'Parametric OLS yields a significant coefficient but imposes")
    print("       linearity; CausalForestDML reveals the underlying threshold")
    print("       structure [cite quartile results Table 7].'")

print("\n" + "=" * 65)
print("✅ HOÀN THÀNH – 06_parametric_benchmark.py")
print("=" * 65)
print(f"  OUTPUT: results/tables/parametric_benchmark.csv")
print()
print("  BƯỚC TIẾP:")
print("  1. Copy bảng CSV vào chat với Claude")
print("  2. Claude sẽ format thành Appendix Table và viết đoạn Section 6.3")
print("=" * 65)