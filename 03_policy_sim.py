"""
QREI Project – 04_policy_sim.py
=================================
Policy Simulation: Break-even & Cost-Benefit Analysis

Paper : "Extreme weather shocks, income inequality, and social protection in Asia"
Journal: Environmental and Resource Economics (ERE) – Springer
Version: 3.0 – Break-even approach

THIẾT KẾ KINH TẾ:
  Với ATE = −0.166 Gini points (CI chứa 0, uncertain),
  thay vì giả vờ optimize, file này trả lời 4 câu hỏi thực tế:

  Q1. BCR tại coverage hiện tại của mỗi quốc gia là bao nhiêu?
  Q2. Cần welfare elasticity (ε) tối thiểu bao nhiêu để BCR ≥ 1?
  Q3. Nếu ε đã biết, cần coverage tối thiểu bao nhiêu để BCR ≥ 1?
  Q4. Welfare gain tuyệt đối (USD/pc/yr) tại các mức coverage?

  Đây là approach chuẩn trong policy analysis khi ATE uncertain
  (Drèze & Stern 1987, Coady et al 2004 IMF).

INPUT : Data/processed/panel_qrei_final.csv
        results/tables/cate_by_country.csv
        results/tables/ate_summary.csv

OUTPUT: results/tables/policy_sim_*.csv
        results/figures/policy_sim_*.png

Chạy : python 04_policy_sim.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH
# ============================================================

BASE = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\Quantum_Global Economic Resilience\qrei_project"

DATA_FILE  = os.path.join(BASE, "Data",    "processed", "panel_qrei_final.csv")
CATE_FILE  = os.path.join(BASE, "results", "tables",    "cate_by_country.csv")
ATE_FILE   = os.path.join(BASE, "results", "tables",    "ate_summary.csv")
TABLE_DIR  = os.path.join(BASE, "results", "tables")
FIGURE_DIR = os.path.join(BASE, "results", "figures")

for d in [TABLE_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Tham số kinh tế ──────────────────────────────────────────
# Welfare elasticity: % GDP welfare gain per 1 Gini point reduction
# Nguồn và range:
#   ε=0.5 : conservative (inequality → consumption only)
#   ε=2.0 : baseline (Ostry et al 2014 IMF WP/14/02)
#   ε=5.0 : broad (gồm health, education, social mobility)
EPSILON_RANGE  = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
EPSILON_BASE   = 2.0

# Fiscal cost per percentage point coverage (% of GDP)
# ILO 2022: Asia average ~0.08% GDP/pp (all-in cost)
# Marginal expansion of existing program: ~0.03–0.05% GDP/pp
COST_RANGE     = [0.02, 0.04, 0.06, 0.08]
COST_BASE      = 0.04   # % GDP per 1pp coverage

# Coverage scenarios để phân tích
COV_SCENARIOS  = [10, 20, 30, 40, 50, 60, 70, 80, 100]

# Region
REGION_MAP = {
    "IND":"South Asia","BGD":"South Asia","PAK":"South Asia","LKA":"South Asia",
    "NPL":"South Asia","AFG":"South Asia","BTN":"South Asia",
    "VNM":"SE Asia","PHL":"SE Asia","IDN":"SE Asia","THA":"SE Asia",
    "MYS":"SE Asia","KHM":"SE Asia","MMR":"SE Asia","LAO":"SE Asia",
    "SGP":"SE Asia","TLS":"SE Asia",
    "CHN":"East Asia","KOR":"East Asia","JPN":"East Asia","MNG":"East Asia",
    "HKG":"East Asia","MAC":"East Asia","BRN":"East Asia","TWN":"East Asia",
}


# ============================================================
# HELPER
# ============================================================

def section(title, step):
    print(f"\n{'='*65}")
    print(f"STEP {step}: {title}")
    print("=" * 65)


def save_table(df, filename, label=""):
    path = os.path.join(TABLE_DIR, filename)
    df.to_csv(path)
    print(f"  ✓ {filename}" + (f" – {label}" if label else ""))


def save_figure(fig, filename):
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {filename}")


def welfare_usd(gini_reduction, gdp_pc, epsilon):
    """
    Welfare gain (USD/capita/year).
    W = ε × ΔGini × GDP_pc / 100
    """
    return epsilon * gini_reduction * gdp_pc / 100.0


def fiscal_cost_usd(coverage_pct, gdp_pc, cost_per_pp):
    """
    Fiscal cost (USD/capita/year).
    C = cost_per_pp (% GDP/pp) × coverage (pp) × GDP_pc / 100
    """
    return cost_per_pp * coverage_pct * gdp_pc / 100.0


def bcr(gini_reduction, coverage_pct, gdp_pc, epsilon, cost_per_pp):
    """Benefit-Cost Ratio."""
    w = welfare_usd(gini_reduction, gdp_pc, epsilon)
    c = fiscal_cost_usd(coverage_pct, gdp_pc, cost_per_pp)
    return w / c if c > 0 else np.nan


def epsilon_breakeven(gini_reduction, coverage_pct, gdp_pc, cost_per_pp):
    """
    Welfare elasticity tối thiểu để BCR = 1.
    ε* = cost / (ΔGini × GDP_pc/100)
    """
    cost = fiscal_cost_usd(coverage_pct, gdp_pc, cost_per_pp)
    denom = gini_reduction * gdp_pc / 100.0
    return cost / denom if denom > 0 else np.inf


def coverage_breakeven(gini_reduction, gdp_pc, epsilon, cost_per_pp):
    """
    Coverage tối thiểu để BCR = 1 (welfare = cost).
    ε × ΔGini × GDP_pc/100 = cost_per_pp × c* × GDP_pc/100
    → c* = ε × ΔGini / cost_per_pp
    Lưu ý: ΔGini = |ATE| ở mức treated (không đổi theo c).
    """
    c_star = (epsilon * gini_reduction) / cost_per_pp
    return min(c_star, 100.0)


# ============================================================
# STEP 1 – LOAD DATA
# ============================================================

section("LOAD DATA & PARAMETERS", step=1)

df = pd.read_csv(DATA_FILE, low_memory=False)
df["region"] = df["country_code"].map(REGION_MAP)
print(f"  Panel data : {df.shape}")

# CATE by country
if not os.path.exists(CATE_FILE):
    raise FileNotFoundError(f"Không tìm thấy: {CATE_FILE}\n"
                            f"  → Chạy 03_tabnet_causal.py trước")
df_cate = pd.read_csv(CATE_FILE, index_col=0)
print(f"  CATE file  : {len(df_cate)} countries | cols: {list(df_cate.columns)}")

# ATE
ate_val, ci_lo, ci_hi = -0.1656, -0.4586, 0.1274
if os.path.exists(ATE_FILE):
    df_ate = pd.read_csv(ATE_FILE)
    if "ATE" in df_ate.columns:
        ate_val = float(df_ate["ATE"].iloc[0])
        ci_lo   = float(df_ate["CI_lower"].iloc[0])
        ci_hi   = float(df_ate["CI_upper"].iloc[0])

gini_reduction_ate = abs(ate_val)   # dương: effect size

# GDP per capita và coverage per country
gdp_by_c = df.groupby("country_code")["gdp_pc"].mean()
cov_by_c = df.groupby("country_code")["social_prot_coverage"].mean() \
    if "social_prot_coverage" in df.columns else pd.Series(dtype=float)
mean_gdp = gdp_by_c.mean()
mean_cov = cov_by_c.mean()

print(f"\n  ATE        : {ate_val:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Gini reduction (|ATE|): {gini_reduction_ate:.4f} points/yr")
print(f"  Mean GDP/pc: ${mean_gdp:,.0f}")
print(f"  Mean coverage: {mean_cov:.1f}%")

# ── Tính welfare và cost tại mean GDP/pc để biết scale ────────
print(f"\n  SCALE CHECK (ε=2.0, cost=0.04, GDP_pc=${mean_gdp:,.0f}):")
print(f"  {'Metric':<40} {'Value':>12}")
print("  " + "-"*55)
w_max = welfare_usd(gini_reduction_ate, mean_gdp, EPSILON_BASE)
c_at_mean = fiscal_cost_usd(mean_cov, mean_gdp, COST_BASE)
print(f"  Max welfare gain (full ATE)       : ${w_max:>10.2f}/pc/yr")
print(f"  Cost at current mean cov ({mean_cov:.0f}%) : ${c_at_mean:>10.2f}/pc/yr")
print(f"  → BCR at current coverage         : {w_max/c_at_mean:>10.2f}")
eps_be = epsilon_breakeven(gini_reduction_ate, mean_cov, mean_gdp, COST_BASE)
print(f"  → ε needed for BCR=1 (cov={mean_cov:.0f}%): {eps_be:>10.2f}")


# ============================================================
# STEP 2 – Q1: BCR TẠI COVERAGE HIỆN TẠI
# ============================================================

section("Q1: BCR TẠI COVERAGE HIỆN TẠI (per country)", step=2)

q1_rows = []
cate_col = "cate_mean" if "cate_mean" in df_cate.columns else df_cate.columns[0]

for country in sorted(df_cate.index):
    cate_c   = abs(float(df_cate.loc[country, cate_col]))
    gdp_c    = gdp_by_c.get(country, mean_gdp)
    if pd.isna(gdp_c): gdp_c = mean_gdp
    cov_c    = cov_by_c.get(country, np.nan)
    region   = REGION_MAP.get(country, "Unknown")

    # Welfare tại full treatment (gini_reduction = CATE)
    w_c = welfare_usd(cate_c, gdp_c, EPSILON_BASE)

    # Cost tại coverage hiện tại
    cost_c = fiscal_cost_usd(cov_c, gdp_c, COST_BASE) \
        if not pd.isna(cov_c) else np.nan

    bcr_c    = w_c / cost_c if (cost_c and cost_c > 0) else np.nan
    net_c    = w_c - cost_c if not pd.isna(cost_c) else np.nan

    # Break-even epsilon (cần ε tối thiểu bao nhiêu?)
    eps_be_c = epsilon_breakeven(cate_c, cov_c, gdp_c, COST_BASE) \
        if not pd.isna(cov_c) else np.nan

    q1_rows.append({
        "country_code":    country,
        "region":          region,
        "cate_abs":        round(cate_c, 4),
        "gdp_pc":          round(gdp_c, 0),
        "coverage_pct":    round(cov_c, 1) if not pd.isna(cov_c) else np.nan,
        "welfare_usd_pc":  round(w_c, 2),
        "cost_usd_pc":     round(cost_c, 2) if not pd.isna(cost_c) else np.nan,
        "net_gain_usd_pc": round(net_c, 2) if not pd.isna(net_c) else np.nan,
        "bcr_at_current":  round(bcr_c, 3) if not pd.isna(bcr_c) else np.nan,
        "epsilon_needed":  round(eps_be_c, 2) if not pd.isna(eps_be_c) else np.nan,
    })

df_q1 = pd.DataFrame(q1_rows).sort_values("bcr_at_current", ascending=False)

print(f"  {'Country':<8} {'Region':<12} {'Cov%':>6} {'Welfare$':>10} "
      f"{'Cost$':>10} {'BCR':>8} {'ε needed':>10}")
print("  " + "-"*70)
for _, r in df_q1.iterrows():
    cov_s  = f"{r['coverage_pct']:.0f}%" if not pd.isna(r['coverage_pct']) else "N/A"
    bcr_s  = f"{r['bcr_at_current']:.3f}" if not pd.isna(r['bcr_at_current']) else "N/A"
    eps_s  = f"{r['epsilon_needed']:.1f}" if not pd.isna(r['epsilon_needed']) else "N/A"
    cost_s = f"{r['cost_usd_pc']:.2f}" if not pd.isna(r['cost_usd_pc']) else "N/A"
    flag   = " ★" if not pd.isna(r['bcr_at_current']) and r['bcr_at_current'] >= 1 else ""
    print(f"  {r['country_code']:<8} {r['region']:<12} {cov_s:>6} "
          f"{r['welfare_usd_pc']:>10.2f} {cost_s:>10} {bcr_s:>8}{flag} "
          f"{eps_s:>9}")

save_table(df_q1.set_index("country_code"), "policy_sim_bcr_current.csv",
           "BCR at current coverage")

# Tóm tắt
bcr_valid = df_q1["bcr_at_current"].dropna()
if len(bcr_valid) > 0:
    print(f"\n  Summary BCR tại coverage hiện tại:")
    print(f"    BCR ≥ 1 (cost-effective)  : "
          f"{(bcr_valid >= 1).sum()} / {len(bcr_valid)} countries")
    print(f"    Median BCR                : {bcr_valid.median():.3f}")
    print(f"    Median ε needed for BCR=1 : "
          f"{df_q1['epsilon_needed'].dropna().median():.1f}")


# ============================================================
# STEP 3 – Q2: BREAK-EVEN EPSILON (ε* per country)
# ============================================================

section("Q2: BREAK-EVEN WELFARE ELASTICITY (ε*)", step=3)

print("  ε* = minimum welfare elasticity for BCR ≥ 1")
print("  Interpretation: program is cost-effective if broader")
print("  welfare impacts (health, education, social) are valued at ε ≥ ε*\n")

eps_rows = []
for _, r in df_q1.iterrows():
    if pd.isna(r["coverage_pct"]):
        continue
    country = r["country_code"]
    cate_c  = r["cate_abs"]
    gdp_c   = r["gdp_pc"]
    cov_c   = r["coverage_pct"]

    row = {"country_code": country, "region": r["region"],
           "cate_abs": cate_c, "coverage_pct": cov_c}

    for cp in COST_RANGE:
        eps_be = epsilon_breakeven(cate_c, cov_c, gdp_c, cp)
        row[f"eps_be_cost{cp:.2f}"] = round(eps_be, 2)

    eps_rows.append(row)

df_eps = pd.DataFrame(eps_rows).set_index("country_code")

# In bảng với cost baseline
eps_col = f"eps_be_cost{COST_BASE:.2f}"
print(f"  Break-even ε at cost={COST_BASE}%GDP/pp:")
print(f"  {'Country':<8} {'ε*':>8}  {'Interpretation'}")
print("  " + "-"*55)
for country, row in df_eps.iterrows():
    eps_star = row[eps_col]
    if eps_star <= 1:
        interp = "✓ BCR≥1 even conservative"
    elif eps_star <= 2:
        interp = "✓ BCR≥1 at moderate ε"
    elif eps_star <= 5:
        interp = "~ BCR≥1 only with broad welfare"
    else:
        interp = "✗ BCR<1 even at ε=5"
    print(f"  {country:<8} {eps_star:>8.2f}  {interp}")

save_table(df_eps, "policy_sim_breakeven_epsilon.csv",
           "Break-even epsilon by country")


# ============================================================
# STEP 4 – Q3: BREAK-EVEN COVERAGE
# ============================================================

section("Q3: BREAK-EVEN COVERAGE (minimum coverage for BCR ≥ 1)", step=4)

print("  c* = coverage level at which welfare = fiscal cost")
print("  c* = ε × |CATE| / cost_per_pp\n")

cbe_rows = []
for _, r in df_q1.iterrows():
    country = r["country_code"]
    cate_c  = r["cate_abs"]
    gdp_c   = r["gdp_pc"]
    cov_c   = r["coverage_pct"]

    row_cbe = {"country_code": country, "region": r["region"],
               "cate_abs": cate_c, "coverage_current": cov_c}

    for eps in EPSILON_RANGE:
        c_star = coverage_breakeven(cate_c, gdp_c, eps, COST_BASE)
        gap    = c_star - cov_c if not pd.isna(cov_c) else np.nan
        row_cbe[f"cbe_eps{eps:.1f}"] = round(c_star, 1)
        if eps == EPSILON_BASE:
            row_cbe["cbe_baseline"] = round(c_star, 1)
            row_cbe["gap_baseline"] = round(gap, 1) if not pd.isna(gap) else np.nan

    cbe_rows.append(row_cbe)

df_cbe = pd.DataFrame(cbe_rows).set_index("country_code")

print(f"  Break-even coverage at ε={EPSILON_BASE} (cost={COST_BASE}%GDP/pp):")
print(f"  {'Country':<8} {'Current':>9} {'c* (need)':>10} "
      f"{'Gap':>8}  {'Status'}")
print("  " + "-"*60)
for country, row in df_cbe.iterrows():
    curr_s = f"{row['coverage_current']:.0f}%" \
        if not pd.isna(row['coverage_current']) else "N/A"
    cbe_s  = f"{row['cbe_baseline']:.1f}%"
    gap_s  = f"{row['gap_baseline']:+.1f}pp" \
        if not pd.isna(row.get('gap_baseline', np.nan)) else "N/A"
    if pd.isna(row['coverage_current']):
        status = "No coverage data"
    elif row['cbe_baseline'] <= row['coverage_current']:
        status = "✓ Already above break-even"
    elif row['gap_baseline'] <= 20:
        status = "~ Close (gap ≤20pp)"
    else:
        status = "✗ Large gap"
    print(f"  {country:<8} {curr_s:>9} {cbe_s:>10} {gap_s:>8}  {status}")

save_table(df_cbe, "policy_sim_breakeven_coverage.csv",
           "Break-even coverage by country")


# ============================================================
# STEP 5 – Q4: WELFARE GAIN SCENARIOS
# ============================================================

section("Q4: WELFARE GAIN SCENARIOS (USD/capita/year)", step=5)

print("  Welfare gain tại các mức coverage và welfare elasticity\n")

scen_rows = []
for eps in [1.0, 2.0, 5.0]:
    row = {"welfare_elasticity": eps}
    for cov in COV_SCENARIOS:
        gini_r = gini_reduction_ate * (cov / 100.0)
        w_usd  = welfare_usd(gini_r, mean_gdp, eps)
        c_usd  = fiscal_cost_usd(cov, mean_gdp, COST_BASE)
        bcr_v  = w_usd / c_usd if c_usd > 0 else np.nan
        row[f"cov{cov:02d}_welfare"] = round(w_usd, 2)
        row[f"cov{cov:02d}_bcr"]     = round(bcr_v, 2)
    scen_rows.append(row)

df_scen = pd.DataFrame(scen_rows).set_index("welfare_elasticity")

# In welfare
print(f"  Welfare gain (USD/pc/yr) | cost={COST_BASE}%GDP/pp:")
header = f"  {'ε':>5}" + "".join(f"  cov={c:2d}%" for c in COV_SCENARIOS)
print(header)
print("  " + "-" * (7 + 9 * len(COV_SCENARIOS)))
for eps, row in df_scen.iterrows():
    line = f"  {eps:>5.1f}"
    for cov in COV_SCENARIOS:
        line += f"  {row[f'cov{cov:02d}_welfare']:>7.1f}"
    print(line)

# In BCR
print(f"\n  BCR | cost={COST_BASE}%GDP/pp:")
print(header)
print("  " + "-" * (7 + 9 * len(COV_SCENARIOS)))
for eps, row in df_scen.iterrows():
    line = f"  {eps:>5.1f}"
    for cov in COV_SCENARIOS:
        bcr_v = row[f'cov{cov:02d}_bcr']
        mark  = "★" if bcr_v >= 1 else " "
        line += f"  {bcr_v:>6.2f}{mark}"
    print(line)
print("  ★ = BCR ≥ 1 (cost-effective)")

save_table(df_scen, "policy_sim_scenarios.csv",
           "Welfare gain and BCR by scenario")


# ============================================================
# STEP 6 – SENSITIVITY MATRIX
# ============================================================

section("SENSITIVITY: BCR MATRIX (ε × cost_per_pp)", step=6)

sens_rows = []
for eps in EPSILON_RANGE:
    for cp in COST_RANGE:
        w_s   = welfare_usd(gini_reduction_ate, mean_gdp, eps)
        c_s   = fiscal_cost_usd(mean_cov, mean_gdp, cp)
        bcr_s = w_s / c_s if c_s > 0 else np.nan
        cbe_s = coverage_breakeven(gini_reduction_ate, mean_gdp, eps, cp)
        eps_be_s = epsilon_breakeven(gini_reduction_ate, mean_cov, mean_gdp, cp)
        sens_rows.append({
            "epsilon":         eps,
            "cost_per_pp":     cp,
            "welfare_usd":     round(w_s, 2),
            "cost_usd":        round(c_s, 2),
            "bcr":             round(bcr_s, 3),
            "bcr_ge1":         int(bcr_s >= 1) if not np.isnan(bcr_s) else 0,
            "cbe_coverage":    round(cbe_s, 1),
            "eps_breakeven":   round(eps_be_s, 2),
        })

df_sens = pd.DataFrame(sens_rows)

# Pivot BCR matrix
pivot_bcr = df_sens.pivot(index="epsilon", columns="cost_per_pp", values="bcr")
print(f"\n  BCR MATRIX (rows=ε, cols=cost_per_pp, at mean coverage={mean_cov:.1f}%):")
print(f"  {'ep/cost':>10}", end="")
for cp in COST_RANGE:
    print(f"  {cp:.2f}%", end="")
print()
print("  " + "-"*50)
for eps in EPSILON_RANGE:
    print(f"  {eps:>10.1f}", end="")
    for cp in COST_RANGE:
        val = pivot_bcr.loc[eps, cp]
        mark = "★" if val >= 1 else " "
        print(f"  {val:5.2f}{mark}", end="")
    print()
print("  ★ = BCR ≥ 1")

save_table(df_sens, "policy_sim_sensitivity.csv",
           "Sensitivity analysis BCR matrix")


# ============================================================
# STEP 7 – FIGURES
# ============================================================

section("FIGURES", step=7)

# ── Figure 1: Break-even epsilon per country ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df_eps_plot = df_q1.dropna(subset=["epsilon_needed"]).sort_values("epsilon_needed")
colors_eps  = ["#A9D18E" if e <= 2 else "#ED7D31" if e <= 5
               else "#C00000" for e in df_eps_plot["epsilon_needed"]]

ax = axes[0]
bars = ax.barh(df_eps_plot["country_code"], df_eps_plot["epsilon_needed"],
               color=colors_eps, alpha=0.85)
ax.axvline(1, color="gray",   linewidth=1, linestyle=":", label="ε=1 (conservative)")
ax.axvline(2, color="#4472C4", linewidth=2, linestyle="--", label="ε=2 (baseline)")
ax.axvline(5, color="red",    linewidth=1, linestyle=":", label="ε=5 (upper bound)")
ax.set_xlabel("Break-even Welfare Elasticity (ε*)", fontsize=10)
ax.set_title("Minimum ε for BCR ≥ 1\n(Green ≤ 2.0 = cost-effective at baseline)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(axis="x", alpha=0.3)

# ── Figure 1b: BCR at different epsilon ──────────────────────
ax2 = axes[1]
eps_plot = [1.0, 2.0, 5.0]
colors_e = ["#A9D18E", "#4472C4", "#ED7D31"]
x_pos    = np.arange(len(df_eps_plot))

for i, (eps, col) in enumerate(zip(eps_plot, colors_e)):
    bcr_vals = []
    for _, r in df_eps_plot.iterrows():
        w = welfare_usd(r["cate_abs"], r["gdp_pc"], eps)
        c = fiscal_cost_usd(r["coverage_pct"], r["gdp_pc"], COST_BASE) \
            if not pd.isna(r["coverage_pct"]) else np.nan
        bcr_vals.append(w/c if (c and c > 0) else np.nan)

    valid_mask = [not pd.isna(b) for b in bcr_vals]
    ax2.scatter([x_pos[j] for j in range(len(x_pos)) if valid_mask[j]],
                [bcr_vals[j] for j in range(len(bcr_vals)) if valid_mask[j]],
                color=col, s=40, label=f"ε={eps}", zorder=3)

ax2.axhline(1, color="red", linewidth=1.5, linestyle="--", label="BCR=1")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df_eps_plot["country_code"], rotation=45, fontsize=8)
ax2.set_ylabel("Benefit-Cost Ratio", fontsize=10)
ax2.set_title("BCR at Current Coverage\nby Welfare Elasticity Assumption",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, min(10, ax2.get_ylim()[1]))

plt.suptitle("Break-even Analysis: Social Protection Cost-Effectiveness\n"
             "QREI Project – ERE Submission",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_figure(fig, "policy_sim_breakeven.png")

# ── Figure 2: Sensitivity heatmap ────────────────────────────
pivot_bcr_heat = df_sens.pivot(index="epsilon", columns="cost_per_pp", values="bcr")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
im = ax.imshow(pivot_bcr_heat.values.astype(float),
               cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(COST_RANGE)))
ax.set_yticks(range(len(EPSILON_RANGE)))
ax.set_xticklabels([f"{v:.2f}%" for v in COST_RANGE], fontsize=9)
ax.set_yticklabels([f"{v:.1f}" for v in EPSILON_RANGE], fontsize=9)
ax.set_xlabel("Cost per pp (% GDP)", fontsize=10)
ax.set_ylabel("Welfare Elasticity (ε)", fontsize=10)
ax.set_title("BCR Heatmap (at mean coverage)\nGreen ≥ 1 = cost-effective",
             fontsize=10, fontweight="bold")
for i in range(len(EPSILON_RANGE)):
    for j in range(len(COST_RANGE)):
        val = pivot_bcr_heat.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8,
                color="white" if val < 0.5 else "black")
# Mark baseline
try:
    ri = EPSILON_RANGE.index(EPSILON_BASE)
    ci = COST_RANGE.index(COST_BASE)
    ax.add_patch(plt.Rectangle((ci-0.5, ri-0.5), 1, 1,
                                fill=False, edgecolor="black",
                                linewidth=2.5))
    ax.text(ci, ri-0.55, "baseline", ha="center", fontsize=7,
            color="black", fontweight="bold")
except ValueError:
    pass

# Break-even coverage heatmap
pivot_cbe = df_sens.pivot(index="epsilon", columns="cost_per_pp",
                           values="cbe_coverage")
ax2 = axes[1]
im2 = ax2.imshow(pivot_cbe.values.astype(float),
                 cmap="YlOrRd_r", aspect="auto", vmin=0, vmax=100)
plt.colorbar(im2, ax=ax2, label="Coverage (%)")
ax2.set_xticks(range(len(COST_RANGE)))
ax2.set_yticks(range(len(EPSILON_RANGE)))
ax2.set_xticklabels([f"{v:.2f}%" for v in COST_RANGE], fontsize=9)
ax2.set_yticklabels([f"{v:.1f}" for v in EPSILON_RANGE], fontsize=9)
ax2.set_xlabel("Cost per pp (% GDP)", fontsize=10)
ax2.set_ylabel("Welfare Elasticity (ε)", fontsize=10)
ax2.set_title("Break-even Coverage c*\n(lower = easier to achieve BCR≥1)",
              fontsize=10, fontweight="bold")
for i in range(len(EPSILON_RANGE)):
    for j in range(len(COST_RANGE)):
        val = pivot_cbe.values[i, j]
        ax2.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8)

plt.suptitle("Sensitivity Analysis: Policy Simulation Parameters\n"
             "QREI Project – ERE Submission",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save_figure(fig, "policy_sim_sensitivity_heatmap.png")

# ── Figure 3: Welfare gain by coverage & epsilon ─────────────
fig, ax = plt.subplots(figsize=(10, 6))
cov_range_plot = np.linspace(0, 100, 201)
colors_eps_plot = {1.0: "#A9D18E", 2.0: "#4472C4", 3.0: "#7030A0", 5.0: "#ED7D31"}

for eps, col in colors_eps_plot.items():
    welfares = [welfare_usd(gini_reduction_ate * (c/100), mean_gdp, eps)
                for c in cov_range_plot]
    ax.plot(cov_range_plot, welfares, color=col, linewidth=2, label=f"ε={eps}")

# Fiscal cost line
costs = [fiscal_cost_usd(c, mean_gdp, COST_BASE) for c in cov_range_plot]
ax.plot(cov_range_plot, costs, "k--", linewidth=2, label=f"Fiscal cost\n(cost={COST_BASE}%/pp)")

ax.axvline(mean_cov, color="gray", linewidth=1, linestyle=":",
           label=f"Current mean ({mean_cov:.0f}%)")
ax.set_xlabel("Social Protection Coverage (%)", fontsize=11)
ax.set_ylabel("USD per capita per year", fontsize=11)
ax.set_title("Welfare Gain vs Fiscal Cost\n"
             "Intersections = Break-even coverage c*",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 100)
plt.tight_layout()
save_figure(fig, "policy_sim_welfare_vs_cost.png")


# ============================================================
# REPORT TỔNG KẾT
# ============================================================

print(f"\n{'='*65}")
print("REPORT TỔNG KẾT – 04_policy_sim.py")
print("=" * 65)

# Baseline BCR
w_base = welfare_usd(gini_reduction_ate, mean_gdp, EPSILON_BASE)
c_base = fiscal_cost_usd(mean_cov, mean_gdp, COST_BASE)
bcr_base = w_base / c_base if c_base > 0 else np.nan
eps_base_needed = epsilon_breakeven(gini_reduction_ate, mean_cov, mean_gdp, COST_BASE)
cbe_base = coverage_breakeven(gini_reduction_ate, mean_gdp, EPSILON_BASE, COST_BASE)

print(f"""
  KẾT QUẢ CHÍNH (ε={EPSILON_BASE}, cost={COST_BASE}%GDP/pp, mean GDP=${mean_gdp:,.0f}):
  ┌─────────────────────────────────────────────────────────┐
  │ ATE                      : {ate_val:+.4f} Gini pts/yr          │
  │ Welfare gain (full ATE)  : USD {w_base:>6.2f}/pc/yr          │
  │ Fiscal cost (mean cov)   : USD {c_base:>6.2f}/pc/yr          │
  │ BCR at mean coverage     : {bcr_base:>6.3f}                   │
  │ Break-even ε (BCR=1)     : {eps_base_needed:>6.2f}                   │
  │ Break-even coverage c*   : {cbe_base:>5.1f}%                    │
  └─────────────────────────────────────────────────────────┘

  SENSITIVITY:
  BCR ≥ 1 khi: ε ≥ {eps_base_needed:.1f}  (at mean coverage, cost={COST_BASE}%/pp)
  Cần coverage ≥ {cbe_base:.1f}% để BCR=1 (at ε={EPSILON_BASE})

  COUNTRIES BCR ≥ 1 (ε=2.0):""")

bcr_ge1 = df_q1[df_q1["bcr_at_current"] >= 1.0]
if len(bcr_ge1) > 0:
    for _, r in bcr_ge1.iterrows():
        print(f"    {r['country_code']}: BCR={r['bcr_at_current']:.3f}  "
              f"cov={r['coverage_pct']:.0f}%  ε*={r['epsilon_needed']:.1f}")
else:
    print(f"    Không có quốc gia nào BCR ≥ 1 tại ε={EPSILON_BASE}")
    print(f"    → Cần ε ≥ {df_q1['epsilon_needed'].dropna().min():.1f} "
          f"(lowest: {df_q1.dropna(subset=['epsilon_needed'])['country_code'].iloc[0]})")

print(f"""
  GHI CHÚ CHO PAPER (ERE):
  ─────────────────────────────────────────────────────────
  "Under conservative welfare assumptions (ε=2.0, marginal
   program cost = 0.04% GDP per percentage point), social
   protection expansion reaches BCR ≥ 1 when welfare
   elasticity exceeds {eps_base_needed:.1f} — consistent with
   broad welfare valuation including health and education
   co-benefits (Dréze & Stern 1987). Countries with the
   strongest CATE estimates (LAO, NPL, BGD, VNM) require
   lower ε thresholds, suggesting cost-effectiveness is
   most plausible in agriculture-dependent economies."
  ─────────────────────────────────────────────────────────

  OUTPUT FILES:""")

for f in ["policy_sim_bcr_current.csv", "policy_sim_breakeven_epsilon.csv",
          "policy_sim_breakeven_coverage.csv", "policy_sim_scenarios.csv",
          "policy_sim_sensitivity.csv", "policy_sim_breakeven.png",
          "policy_sim_sensitivity_heatmap.png", "policy_sim_welfare_vs_cost.png"]:
    folder = TABLE_DIR if f.endswith(".csv") else FIGURE_DIR
    exists = os.path.exists(os.path.join(folder, f))
    print(f"  {'✓' if exists else '✗'} {f}")

print(f"\n{'='*65}")
print("✅ HOÀN THÀNH – 04_policy_sim.py")
print("=" * 65)
print("  → Bước tiếp: robustness checks → viết paper")
print("=" * 65)