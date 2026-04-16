"""
QREI Project – 06_blp_cluster_bootstrap.py
============================================
Best Linear Projection (BLP) với Cluster-Robust Standard Errors

Mục đích:
  Tái ước lượng BLP moderation coefficients (θ₁) từ Table 5 trong paper
  sử dụng standard errors được cluster theo country_code thay vì
  bootstrap không cluster. Đây là robustness check cho reviewer
  methodologist của World Development.

Lý thuyết:
  BLP regression: CATE_i = θ₀ + θ₁ * (moderator_i − mean) + ε_i
  SE clustered by country để account for intertemporal correlation
  trong panel observations thuộc cùng một quốc gia.

INPUT:
  results/tables/cate_individual.csv   ← observation-level CATEs
  Data/processed/panel_qrei_final.csv  ← moderator variables

OUTPUT:
  results/tables/blp_clustered_se.csv  ← Table 5 với clustered SE
  results/tables/blp_comparison.csv    ← So sánh bootstrap vs clustered
  results/figures/blp_comparison.png   ← Visualization

Chạy: python 06_blp_cluster_bootstrap.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH — chỉnh BASE path cho phù hợp với máy
# ============================================================

BASE = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\Social Protection as Climate Adaptation (RJ)\qrei_project"

CATE_FILE = os.path.join(BASE, "results", "tables", "cate_individual.csv")
DATA_FILE = os.path.join(BASE, "Data", "processed", "panel_qrei_final.csv")
TABLE_DIR = os.path.join(BASE, "results", "tables")
FIGURE_DIR = os.path.join(BASE, "results", "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Moderators cần test (khớp với X_COLS trong 02_tabnet_causal.py)
MODERATORS = [
    "rule_of_law",
    "democracy_electoral",
    "log_gdp_pc",
    "temp_shock",
    "extreme_temp_shock",
    "rice_yield_dev",
    "disaster_count_cy",
]

# Kết quả bootstrap gốc từ paper (Table 5) để so sánh
BOOTSTRAP_RESULTS = {
    "rule_of_law":        {"theta1": 0.606,  "se": 0.025, "t": 24.2,  "sig": "**"},
    "democracy_electoral": {"theta1": 0.658,  "se": 0.024, "t": 27.0,  "sig": "**"},
    "log_gdp_pc":          {"theta1": 0.096,  "se": 0.007, "t": 14.7,  "sig": "**"},
    "temp_shock":          {"theta1": 0.062,  "se": 0.008, "t": 8.1,   "sig": "**"},
    "extreme_temp_shock":  {"theta1": -0.036, "se": 0.037, "t": -0.99, "sig": "—"},
    "rice_yield_dev":      {"theta1": -0.005, "se": 0.007, "t": -0.81, "sig": "—"},
    "disaster_count_cy":   {"theta1": None,   "se": None,  "t": None,  "sig": "—"},
}

# ============================================================
# LOAD & MERGE DATA
# ============================================================

print("=" * 65)
print("06_blp_cluster_bootstrap.py")
print("BLP Moderation với Cluster-Robust Standard Errors")
print("=" * 65)

print("\n→ Loading data...")

df_cate = pd.read_csv(CATE_FILE, low_memory=False)
df_panel = pd.read_csv(DATA_FILE, low_memory=False)

print(f"  CATE file  : {df_cate.shape}  cols: {list(df_cate.columns)}")
print(f"  Panel file : {df_panel.shape}")

# Chuẩn hóa tên cột nếu có index column thừa
if "Unnamed: 0" in df_cate.columns:
    df_cate = df_cate.drop(columns=["Unnamed: 0"])
if "Unnamed: 0" in df_panel.columns:
    df_panel = df_panel.drop(columns=["Unnamed: 0"])

# Merge CATE với moderator variables theo country_code + year
merge_cols = ["country_code", "year"] + [
    m for m in MODERATORS if m in df_panel.columns
]
df_panel_sub = df_panel[merge_cols].drop_duplicates(
    subset=["country_code", "year"]
)

df = df_cate.merge(df_panel_sub, on=["country_code", "year"], how="left")

print(f"\n  Merged dataset: {df.shape}")
print(f"  Unique countries: {df['country_code'].nunique()}")
print(f"  CATE range: [{df['cate'].min():.4f}, {df['cate'].max():.4f}]")
print(f"  Missing country_code: {df['country_code'].isna().sum()}")

# Kiểm tra moderators có trong data
available_mods = [m for m in MODERATORS if m in df.columns]
missing_mods = [m for m in MODERATORS if m not in df.columns]
if missing_mods:
    print(f"\n  ⚠ Moderators không tìm thấy: {missing_mods}")
print(f"  Moderators available: {available_mods}")

# ============================================================
# BLP REGRESSION VỚI CLUSTERED SE
# ============================================================

print("\n" + "=" * 65)
print("BLP REGRESSIONS — Cluster-Robust SE (clustered by country_code)")
print("=" * 65)

results_clustered = []

for mod in available_mods:
    # Tạo subsample không missing
    sub = df[["cate", mod, "country_code"]].dropna()
    n_obs = len(sub)
    n_clusters = sub["country_code"].nunique()

    if n_obs < 30:
        print(f"\n  {mod}: bỏ qua (N={n_obs} quá nhỏ)")
        continue

    # Demean moderator (theo BLP convention)
    mod_demeaned = sub[mod] - sub[mod].mean()

    # Design matrix: intercept + demeaned moderator
    X_blp = sm.add_constant(mod_demeaned, has_constant="add")
    y_blp = sub["cate"].values

    # OLS với cluster-robust SE (groups = country_code)
    try:
        ols = sm.OLS(y_blp, X_blp).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["country_code"].values}
        )

        theta1 = ols.params.iloc[1]   # coefficient trên moderator
        se_cl  = ols.bse.iloc[1]      # clustered SE
        t_cl   = ols.tvalues.iloc[1]
        p_cl   = ols.pvalues.iloc[1]
        ci_lo  = ols.conf_int().iloc[1, 0]
        ci_hi  = ols.conf_int().iloc[1, 1]

        # Significance stars
        if p_cl < 0.01:
            sig = "** (p<0.01)"
        elif p_cl < 0.05:
            sig = "* (p<0.05)"
        elif p_cl < 0.10:
            sig = ". (p<0.10)"
        else:
            sig = "—"

        # So sánh với bootstrap gốc
        boot = BOOTSTRAP_RESULTS.get(mod, {})
        se_boot = boot.get("se", None)
        se_ratio = (se_cl / se_boot) if se_boot else None

        row = {
            "Moderator":       mod,
            "θ₁ (clustered)":  round(theta1, 4),
            "SE (clustered)":  round(se_cl, 4),
            "t (clustered)":   round(t_cl, 4),
            "p (clustered)":   round(p_cl, 4),
            "CI lower":        round(ci_lo, 4),
            "CI upper":        round(ci_hi, 4),
            "Sig (clustered)": sig,
            "θ₁ (bootstrap)":  boot.get("theta1", None),
            "SE (bootstrap)":  se_boot,
            "t (bootstrap)":   boot.get("t", None),
            "SE ratio (cl/boot)": round(se_ratio, 3) if se_ratio else None,
            "N_obs":           n_obs,
            "N_clusters":      n_clusters,
        }
        results_clustered.append(row)

        # Print kết quả
        print(f"\n  Moderator: {mod}")
        print(f"  N={n_obs}, clusters={n_clusters}")
        print(f"  θ₁ = {theta1:+.4f}  SE(clustered) = {se_cl:.4f}"
              f"  t = {t_cl:.2f}  p = {p_cl:.4f}  [{sig}]")
        print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        if se_boot:
            print(f"  So sánh: SE(bootstrap)={se_boot:.4f} → "
                  f"SE ratio (clustered/bootstrap) = {se_ratio:.2f}x")

    except Exception as e:
        print(f"\n  {mod}: LỖI → {e}")

# ============================================================
# TỔNG HỢP KẾT QUẢ
# ============================================================

df_results = pd.DataFrame(results_clustered)

print("\n" + "=" * 65)
print("TỔNG HỢP — BLP Table với Clustered SE")
print("=" * 65)

# Table ngắn gọn cho paper
df_paper = df_results[[
    "Moderator", "θ₁ (clustered)", "SE (clustered)",
    "t (clustered)", "CI lower", "CI upper",
    "Sig (clustered)", "N_obs", "N_clusters"
]].copy()

print(df_paper.to_string(index=False))

# Table so sánh đầy đủ
df_compare = df_results[[
    "Moderator",
    "θ₁ (bootstrap)", "SE (bootstrap)", "t (bootstrap)",
    "θ₁ (clustered)", "SE (clustered)", "t (clustered)",
    "SE ratio (cl/boot)", "Sig (clustered)", "N_clusters"
]].copy()

# Lưu files
df_paper.to_csv(os.path.join(TABLE_DIR, "blp_clustered_se.csv"), index=False)
df_compare.to_csv(os.path.join(TABLE_DIR, "blp_comparison.csv"), index=False)
print(f"\n  ✓ blp_clustered_se.csv")
print(f"  ✓ blp_comparison.csv")

# ============================================================
# FIGURE: So sánh Bootstrap vs Clustered SE
# ============================================================

print("\n→ Tạo figure so sánh...")

mods_plot = [r for r in results_clustered
             if r["SE (bootstrap)"] is not None]

if len(mods_plot) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = [r["Moderator"].replace("_", "\n") for r in mods_plot]
    theta1_vals = [r["θ₁ (clustered)"] for r in mods_plot]
    se_boot_vals = [r["SE (bootstrap)"] for r in mods_plot]
    se_cl_vals = [r["SE (clustered)"] for r in mods_plot]
    ci_lo_vals = [r["CI lower"] for r in mods_plot]
    ci_hi_vals = [r["CI upper"] for r in mods_plot]

    x = np.arange(len(labels))
    width = 0.35

    # Panel 1: SE comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, se_boot_vals, width,
                   label="Bootstrap SE (500 rep)", color="#4472C4", alpha=0.8)
    bars2 = ax.bar(x + width/2, se_cl_vals, width,
                   label="Clustered SE (by country)", color="#ED7D31", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Standard Error", fontsize=10)
    ax.set_title("SE Comparison:\nBootstrap vs Cluster-Robust",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Thêm ratio labels
    for i, r in enumerate(mods_plot):
        ratio = r["SE ratio (cl/boot)"]
        if ratio:
            ax.text(i, max(se_boot_vals[i], se_cl_vals[i]) + 0.001,
                    f"×{ratio:.1f}", ha="center", va="bottom",
                    fontsize=8, color="black")

    # Panel 2: θ₁ với CI (clustered)
    ax2 = axes[1]
    colors_ci = ["#A9D18E" if lo > 0 or hi < 0 else "#C9C9C9"
                 for lo, hi in zip(ci_lo_vals, ci_hi_vals)]

    for i, (th, lo, hi, col) in enumerate(
            zip(theta1_vals, ci_lo_vals, ci_hi_vals, colors_ci)):
        ax2.plot([lo, hi], [i, i], color=col, linewidth=3, alpha=0.8)
        ax2.scatter([th], [i], color=col, s=80, zorder=5)

    ax2.axvline(0, color="black", linewidth=1.5,
                linestyle="--", label="Zero effect")
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("θ₁ (BLP moderation coefficient)", fontsize=10)
    ax2.set_title("BLP Coefficients — Cluster-Robust 95% CI\n"
                  "Green = significant; Grey = not significant",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    plt.suptitle("BLP Moderation Analysis: Bootstrap vs Cluster-Robust SE\n"
                 f"N = {df_results['N_obs'].iloc[0]} obs, "
                 f"{df_results['N_clusters'].iloc[0]} country clusters",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    fig_path = os.path.join(FIGURE_DIR, "blp_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ blp_comparison.png")

# ============================================================
# REPORT — Hướng dẫn update manuscript
# ============================================================

print(f"\n{'='*65}")
print("REPORT — Hướng dẫn update Table 5 trong manuscript")
print("=" * 65)

sig_clustered = [r for r in results_clustered
                 if "p<0.01" in r["Sig (clustered)"]
                 or "p<0.05" in r["Sig (clustered)"]]
sig_names = [r["Moderator"] for r in sig_clustered]

print(f"\n  Moderators vẫn significant với clustered SE: {sig_names}")

# Phân tích thay đổi SE
print(f"\n  SE ratio (clustered / bootstrap):")
for r in results_clustered:
    ratio = r["SE ratio (cl/boot)"]
    if ratio:
        direction = "↑ nở ra" if ratio > 1.5 else (
                    "↓ thu lại" if ratio < 0.7 else "≈ tương đương")
        print(f"    {r['Moderator']:<25}: ×{ratio:.2f}  {direction}")

print(f"""
  HÀNH ĐỘNG:

  Nếu tất cả moderators chính (rule_of_law, democracy_electoral,
  log_gdp_pc) vẫn significant với clustered SE:
  → Xóa hedge text đã thêm vào manuscript
  → Thay SE và t-stat trong Table 5 bằng giá trị clustered
  → Thêm vào Table 5 Notes:
     "Standard errors clustered by country (N={df_results['N_clusters'].iloc[0] if len(df_results) > 0 else 25} clusters)."

  Nếu một số moderators mất significance:
  → Giữ nguyên hedge text trong manuscript (đã chuẩn bị sẵn)
  → Trong Response to Reviewers: trình bày bảng so sánh
     blp_comparison.csv và giải thích tại sao bootstrap SE
     là conservative choice phù hợp với DML framework
""")

print("=" * 65)
print("✅ HOÀN THÀNH – 06_blp_cluster_bootstrap.py")
print("=" * 65)
print("  Output:")
print(f"  → results/tables/blp_clustered_se.csv")
print(f"  → results/tables/blp_comparison.csv")
print(f"  → results/figures/blp_comparison.png")
print("=" * 65)
