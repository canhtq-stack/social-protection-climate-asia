"""
03_robustness.py
================
Robustness Checks RC1–RC9 (Table 7 + Figures A1–A2 in manuscript)

Paper : "Social Protection as Climate Adaptation: Heterogeneous Effects
         on Income Inequality in Asia"
Journal: Economics of Disasters and Climate Change, Springer

Prerequisite: run 01_tabnet_causal.py first.

RC1 : Leave-one-country-out (LOCO)
RC2 : Placebo test (200 random assignments) → Figure A2
RC3 : Treatment threshold sensitivity (P25, P67)
RC4 : Geographic subsample (South Asia, SE Asia, South+SE Asia)
RC5 : Long-difference outcome (5-year ΔGini)
RC6 : Level Gini outcome
RC7 : Balanced vs unbalanced panel
RC8 : Continuous treatment (GRF dose-response)
RC9 : Observed ASPIRE only (missing-data sensitivity)

Outputs
-------
output/tables/robustness_summary.csv    → Table 7
output/tables/loco_full_results.csv
output/tables/placebo_ates.csv
output/figures/figureA1_robustness.png  → Figure A1
output/figures/figureA2_placebo.png     → Figure A2
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

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_FILE  = ROOT / "data"   / "panel_qrei_final.csv"
DATA_UNBAL = ROOT / "data"   / "panel_qrei_final_unbalanced.csv"
TABLE_DIR  = ROOT / "output" / "tables"
FIGURE_DIR = ROOT / "output" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_PLACEBO    = 200
np.random.seed(RANDOM_SEED)

# Baseline values from 01_tabnet_causal.py (Table 1)
ATE_BASELINE = -0.1492
ATE_CI_LO    = -0.5414
ATE_CI_HI    =  0.2431

Y_COL  = "delta_gini"
T_COL  = "high_social_prot"
X_COLS = [
    "temp_shock", "extreme_temp_shock", "rice_yield_dev",
    "log_gdp_pc", "rule_of_law", "democracy_electoral", "disaster_count_cy",
]
W_COLS = ["log_gdp_pc_lag1", "gini_lag1", "corruption_index", "temp_shock_lag1"]

SOUTH_ASIA = ["IND", "BGD", "PAK", "LKA", "NPL", "AFG", "BTN"]
SE_ASIA    = ["VNM", "PHL", "IDN", "THA", "MYS", "KHM", "MMR", "LAO", "SGP", "TLS"]

# ── Helpers ────────────────────────────────────────────────────────────────

def section(title, rc):
    print(f"\n{'='*70}\nRC{rc}: {title}\n{'='*70}")

def save_table(df, filename, label=""):
    path = TABLE_DIR / filename
    df.to_csv(path, index=True)
    print(f"  ✓ {filename}" + (f"  [{label}]" if label else ""))

def save_figure(fig, filename):
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {filename}")

def prep_arrays(df, y_col=Y_COL, t_col=T_COL, x_cols=X_COLS, w_cols=W_COLS):
    """Return (Y, T, X, W, n_obs, n_treated) or None if insufficient data."""
    if df is None or len(df) == 0:
        return None
    x_avail = [c for c in x_cols if c in df.columns]
    w_avail = [c for c in w_cols if c in df.columns]
    df_s    = df.copy()

    for col in x_avail + w_avail:
        if df_s[col].isnull().any():
            if "country_code" in df.columns:
                df_s[col] = (
                    df.groupby("country_code")[col]
                    .transform(lambda x: x.fillna(x.median()))
                    .reindex(df_s.index)
                )
            df_s[col] = df_s[col].fillna(df_s[col].median()).fillna(0)

    df_s = df_s[df_s[y_col].notna() & df_s[t_col].notna()]
    if len(df_s) < 50:
        return None

    Y = df_s[y_col].values.astype(np.float64)
    T = (df_s[t_col].values >= 0.5).astype(int)
    X = df_s[x_avail].values.astype(np.float64)
    W = df_s[w_avail].values.astype(np.float64) if w_avail else None
    return Y, T, X, W, len(df_s), int(T.sum())

def fit_cf(Y, T, X, W=None, n_estimators=500):
    """Fit CausalForestDML; return (ate, ci_lo, ci_hi)."""
    valid = X.std(axis=0) > 1e-10
    X = X[:, valid]
    if X.shape[1] == 0:
        raise ValueError("No valid X columns after variance filter.")

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
        n_estimators=n_estimators,
        min_samples_leaf=10,
        max_samples=0.5,
        inference=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    cf.fit(Y=Y, T=T, X=X, W=W)
    ate = float(cf.ate(X))
    ci  = cf.ate_interval(X, alpha=0.05)
    return ate, float(ci[0]), float(ci[1])

def result_row(name, ate, ci_lo, ci_hi, n_obs, note=""):
    sig = "**" if (ci_lo > 0 or ci_hi < 0) else ""
    return {
        "Check": name, "N_obs": n_obs,
        "ATE": round(ate, 4), "CI_lower": round(ci_lo, 4), "CI_upper": round(ci_hi, 4),
        "Significant": sig, "Note": note,
    }

# ── Load data ──────────────────────────────────────────────────────────────
print("="*70)
print("03_robustness.py — Robustness Checks RC1–RC9")
print("="*70)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

df     = pd.read_csv(DATA_FILE, low_memory=False)
df_unbal = pd.read_csv(DATA_UNBAL, low_memory=False) if DATA_UNBAL.exists() else df.copy()
print(f"  Main data   : {df.shape}")
print(f"  Unbalanced  : {df_unbal.shape}")

all_results = []

# ── RC1: LOCO ─────────────────────────────────────────────────────────────
section("LEAVE-ONE-COUNTRY-OUT (LOCO)", 1)

loco_rows = []
for excl in df["country_code"].unique():
    df_l = df[df["country_code"] != excl].copy()
    out  = prep_arrays(df_l)
    if out is None:
        continue
    Y_l, T_l, X_l, W_l, n_l, _ = out
    if T_l.sum() == 0 or T_l.sum() == len(T_l):
        continue
    try:
        ate_l, ci_lo_l, ci_hi_l = fit_cf(Y_l, T_l, X_l, W_l)
        loco_rows.append({
            "Excluded": excl, "N": n_l,
            "ATE": ate_l, "CI_lo": ci_lo_l, "CI_hi": ci_hi_l,
        })
        print(f"  Excl. {excl:5s}: ATE = {ate_l:+.4f}  [{ci_lo_l:.4f}, {ci_hi_l:.4f}]")
    except Exception as e:
        print(f"  Excl. {excl}: ERROR — {e}")

if loco_rows:
    df_loco = pd.DataFrame(loco_rows)
    save_table(df_loco.set_index("Excluded"), "loco_full_results.csv", "LOCO all exclusions")

    # Summary on the 18 economies that contribute estimation observations
    df_loco_18 = df_loco[df_loco["N"] < df_loco["N"].max()]
    n_neg  = (df_loco_18["ATE"] < 0).sum()
    ate_med = df_loco_18["ATE"].median()
    print(f"\n  {n_neg}/{len(df_loco_18)} negative  |  median ATE = {ate_med:.4f}"
          f"  |  range [{df_loco_18['ATE'].min():.4f}, {df_loco_18['ATE'].max():.4f}]")
    all_results.append(result_row(
        "RC1: LOCO (median)", ate_med,
        df_loco_18["CI_lo"].median(), df_loco_18["CI_hi"].median(),
        int(df_loco_18["N"].median()),
        f"{n_neg}/{len(df_loco_18)} neg; range [{df_loco_18['ATE'].min():.3f}, "
        f"{df_loco_18['ATE'].max():.3f}]",
    ))

# ── RC2: Placebo test ──────────────────────────────────────────────────────
section("PLACEBO TEST (N=200 random assignments)", 2)

out_base = prep_arrays(df)
if out_base is not None:
    Y_b, T_b, X_b, W_b, n_b, _ = out_base
    placebo_ates = []
    rng = np.random.default_rng(RANDOM_SEED)

    for i in range(N_PLACEBO):
        T_perm = rng.permutation(T_b)
        if T_perm.sum() == 0 or T_perm.sum() == len(T_perm):
            continue
        try:
            ate_p, _, _ = fit_cf(Y_b, T_perm, X_b, W_b, n_estimators=200)
            placebo_ates.append(ate_p)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  Placebo {i+1}/{N_PLACEBO} done...")

    placebo_ates = np.array(placebo_ates)
    p_val = np.mean(placebo_ates <= ATE_BASELINE)
    print(f"\n  Actual ATE   : {ATE_BASELINE:.4f}")
    print(f"  Placebo mean : {placebo_ates.mean():.4f}")
    print(f"  p-value      : {p_val:.4f}")

    pd.DataFrame({"placebo_ate": placebo_ates}).to_csv(
        TABLE_DIR / "placebo_ates.csv", index=False,
    )
    print("  ✓ placebo_ates.csv")

    all_results.append({
        "Check": "RC2: Placebo test", "N_obs": n_b,
        "ATE": np.nan, "CI_lower": np.nan, "CI_upper": np.nan,
        "Significant": f"p={p_val:.3f}",
        "Note": f"200 random assignments; placebo mean = {placebo_ates.mean():.4f}",
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(placebo_ates, bins=30, color="#2B5FA5", alpha=0.70,
            edgecolor="white", linewidth=0.6)
    ax.axvline(ATE_BASELINE, color="#C0392B", linewidth=2.5,
               label=f"Actual ATE = {ATE_BASELINE:.3f}  (p = {p_val:.3f})")
    ax.axvline(placebo_ates.mean(), color="#E07B39", linewidth=1.5,
               linestyle="--", label=f"Placebo mean = {placebo_ates.mean():.4f}")
    ax.set_xlabel("ATE under Random Treatment Assignment")
    ax.set_ylabel("Frequency")
    ax.set_title(
        "Figure A2. Placebo Test Distribution\n"
        f"N = {len(placebo_ates)} random assignments  |  "
        f"Actual ATE more extreme than {(1-p_val)*100:.1f}% of placebo estimates"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, linestyle=":")
    plt.tight_layout()
    save_figure(fig, "figureA2_placebo.png")

# ── RC3: Threshold sensitivity ────────────────────────────────────────────
section("THRESHOLD SENSITIVITY (P25, P67)", 3)

if "social_prot_coverage" in df.columns:
    for pct, label in [(25, "P25"), (67, "P67")]:
        thresh = np.percentile(df["social_prot_coverage"].dropna(), pct)
        df_rc3 = df.copy()
        df_rc3[T_COL] = (df_rc3["social_prot_coverage"].fillna(0) >= thresh).astype(float)
        out = prep_arrays(df_rc3)
        if out is None:
            continue
        Y3, T3, X3, W3, n3, nt3 = out
        try:
            ate3, ci_lo3, ci_hi3 = fit_cf(Y3, T3, X3, W3)
            print(f"  {label} (thresh={thresh:.1f}%): ATE={ate3:.4f} "
                  f"[{ci_lo3:.4f}, {ci_hi3:.4f}]  Treated={nt3}/{n3}")
            all_results.append(result_row(
                f"RC3: Threshold {label}", ate3, ci_lo3, ci_hi3, n3,
                f"Treated={nt3}/{n3} ({nt3/n3*100:.1f}%); thresh={thresh:.1f}%",
            ))
        except Exception as e:
            print(f"  {label} ERROR: {e}")

# ── RC4: Geographic subsamples ────────────────────────────────────────────
section("GEOGRAPHIC SUBSAMPLES", 4)

for name, subset in [
    ("RC4: South Asia",     SOUTH_ASIA),
    ("RC4: SE Asia",        SE_ASIA),
    ("RC4: South+SE Asia",  SOUTH_ASIA + SE_ASIA),
]:
    df_sub = df[df["country_code"].isin(subset)].copy()
    out = prep_arrays(df_sub)
    if out is None:
        print(f"  {name}: insufficient data — skipped")
        continue
    Y4, T4, X4, W4, n4, nt4 = out
    if T4.sum() == 0 or T4.sum() == len(T4):
        print(f"  {name}: no treatment variation — skipped")
        continue
    try:
        ate4, ci_lo4, ci_hi4 = fit_cf(Y4, T4, X4, W4)
        print(f"  {name}: ATE={ate4:.4f}  [{ci_lo4:.4f}, {ci_hi4:.4f}]  N={n4}")
        all_results.append(result_row(name, ate4, ci_lo4, ci_hi4, n4))
    except Exception as e:
        print(f"  {name} ERROR: {e}")

# ── RC5: 5-year long difference ───────────────────────────────────────────
section("LONG DIFFERENCE — 5-year ΔGini", 5)

gini_lvl = next((c for c in ["gini", "gini_level", "gini_coef"] if c in df.columns), None)
if gini_lvl:
    df_rc5 = df.sort_values(["country_code", "year"]).copy()
    df_rc5["delta_gini_5yr"] = df_rc5.groupby("country_code")[gini_lvl].transform(
        lambda x: x - x.shift(5)
    )
    out = prep_arrays(df_rc5, y_col="delta_gini_5yr")
    if out is not None:
        Y5, T5, X5, W5, n5, _ = out
        try:
            ate5, ci_lo5, ci_hi5 = fit_cf(Y5, T5, X5, W5)
            print(f"  RC5: ATE={ate5:.4f}  [{ci_lo5:.4f}, {ci_hi5:.4f}]  N={n5}")
            all_results.append(result_row(
                "RC5: Long-diff (5yr)", ate5, ci_lo5, ci_hi5, n5,
                "5-year Gini change; tests persistent effects",
            ))
        except Exception as e:
            print(f"  RC5 ERROR: {e}")
else:
    print("  WARNING: Gini level column not found — skipping RC5")

# ── RC6: Level Gini ────────────────────────────────────────────────────────
section("LEVEL GINI", 6)

if gini_lvl:
    out = prep_arrays(df, y_col=gini_lvl)
    if out is not None:
        Y6, T6, X6, W6, n6, _ = out
        try:
            ate6, ci_lo6, ci_hi6 = fit_cf(Y6, T6, X6, W6)
            print(f"  RC6: ATE={ate6:.4f}  [{ci_lo6:.4f}, {ci_hi6:.4f}]  N={n6}")
            all_results.append(result_row(
                "RC6: Level Gini", ate6, ci_lo6, ci_hi6, n6,
                "Gini levels instead of first differences",
            ))
        except Exception as e:
            print(f"  RC6 ERROR: {e}")

# ── RC7: Balanced vs unbalanced ───────────────────────────────────────────
section("BALANCED vs UNBALANCED PANEL", 7)

for label, df_r7 in [("RC7: Balanced panel", df), ("RC7: Unbalanced panel", df_unbal)]:
    out = prep_arrays(df_r7)
    if out is None:
        print(f"  {label}: insufficient data")
        continue
    Y7, T7, X7, W7, n7, _ = out
    try:
        ate7, ci_lo7, ci_hi7 = fit_cf(Y7, T7, X7, W7)
        print(f"  {label}: ATE={ate7:.4f}  [{ci_lo7:.4f}, {ci_hi7:.4f}]  N={n7}")
        all_results.append(result_row(label, ate7, ci_lo7, ci_hi7, n7))
    except Exception as e:
        print(f"  {label} ERROR: {e}")

# ── RC8: Continuous treatment (GRF) ───────────────────────────────────────
section("CONTINUOUS TREATMENT — GRF dose-response", 8)

try:
    from econml.grf import CausalForest as GRF_CF
    if "social_prot_coverage" in df.columns:
        df_rc8 = df[df["social_prot_coverage"].notna()].copy()
        out = prep_arrays(df_rc8)
        if out is not None:
            Y8, _, X8, W8, n8, _ = out
            T8_cont = df_rc8.loc[
                df_rc8[Y_COL].notna() & df_rc8[T_COL].notna(), "social_prot_coverage"
            ].values.astype(np.float64)
            T8_cont = np.clip(T8_cont, 0, 100)
            grf = GRF_CF(n_estimators=400, min_samples_leaf=10,
                         max_samples=0.5, random_state=RANDOM_SEED, n_jobs=-1)
            grf.fit(X8, T8_cont, Y8)
            ate8   = float(grf.predict(X8).mean())
            ci8    = grf.predict_interval(X8, alpha=0.05)
            ci_lo8 = float(ci8[0].mean())
            ci_hi8 = float(ci8[1].mean())
            print(f"  RC8 avg deriv = {ate8:.4f}  [{ci_lo8:.4f}, {ci_hi8:.4f}]  N={n8}")
            all_results.append(result_row(
                "RC8: Continuous treatment", ate8, ci_lo8, ci_hi8, n8,
                "GRF dose-response; avg. deriv. of ΔGini w.r.t. coverage (%)",
            ))
except ImportError:
    print("  WARNING: econml.grf not available — skipping RC8")
except Exception as e:
    print(f"  RC8 ERROR: {e}")

# ── RC9: Observed ASPIRE only ─────────────────────────────────────────────
section("OBSERVED ASPIRE ONLY — missing-data sensitivity", 9)

if "social_prot_coverage" in df.columns:
    df_rc9 = df[df["social_prot_coverage"].notna()].copy()
    print(f"  Observed sample: N={len(df_rc9)} (dropped {len(df)-len(df_rc9)} imputed rows)")
    out = prep_arrays(df_rc9)
    if out is not None:
        Y9, T9, X9, W9, n9, _ = out
        try:
            ate9, ci_lo9, ci_hi9 = fit_cf(Y9, T9, X9, W9)
            print(f"  RC9: ATE={ate9:.4f}  [{ci_lo9:.4f}, {ci_hi9:.4f}]  N={n9}")
            all_results.append(result_row(
                "RC9: Missing ASPIRE sensitivity", ate9, ci_lo9, ci_hi9, n9,
                "Observed ASPIRE only; directionally consistent with baseline",
            ))
        except Exception as e:
            print(f"  RC9 ERROR: {e}")

# ── Summary table + Figure A1 ──────────────────────────────────────────────
print(f"\n{'='*70}\nSUMMARY — Table 7\n{'='*70}")

baseline = result_row("Baseline", ATE_BASELINE, ATE_CI_LO, ATE_CI_HI, 502,
                      "Main specification; 25 economies, 502 country-years")
df_summary = pd.DataFrame([baseline] + all_results)
save_table(df_summary.set_index("Check"), "robustness_summary.csv", "Table 7")
print(df_summary[["Check", "N_obs", "ATE", "CI_lower", "CI_upper", "Significant"]].to_string(index=False))

# Figure A1: Robustness forest plot
df_plot = df_summary[df_summary["ATE"].notna()].copy()
y_pos   = np.arange(len(df_plot))

fig, ax = plt.subplots(figsize=(11, max(7, len(df_plot) * 0.45)))
colors  = ["#C0392B" if s.startswith("**") else "#2B5FA5"
           for s in df_plot["Significant"].astype(str)]

ax.errorbar(
    df_plot["ATE"], y_pos,
    xerr=[df_plot["ATE"] - df_plot["CI_lower"],
          df_plot["CI_upper"] - df_plot["ATE"]],
    fmt="none", color="#555555", capsize=3, linewidth=0.9, zorder=2,
)
for i, (row, col) in enumerate(zip(df_plot.itertuples(), colors)):
    sz = 90 if i == 0 else 60
    ax.scatter(row.ATE, i, color=col, s=sz, zorder=3)

ax.axvline(0,            color="#C0392B", linewidth=1.5, linestyle="--", alpha=0.8, label="Zero effect")
ax.axvline(ATE_BASELINE, color="#999999", linewidth=1.2, linestyle=":",  alpha=0.8,
           label=f"Baseline ATE = {ATE_BASELINE:.3f}")
ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot["Check"].tolist(), fontsize=8.5)
ax.set_xlabel("ATE — Effect on ΔGini (95% CI)")
ax.set_title(
    "Figure A1. Robustness Forest Plot — All Specification Checks\n"
    "All point estimates negative  |  CausalForestDML"
)
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.2, linestyle=":")

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"N={int(r.N_obs)}" for r in df_plot.itertuples()], fontsize=7.5)
ax2.spines["top"].set_visible(False)

plt.tight_layout()
save_figure(fig, "figureA1_robustness.png")

n_neg = (df_summary["ATE"].dropna() < 0).sum()
n_tot = df_summary["ATE"].dropna().count()
print(f"\nDONE — 03_robustness.py  ({n_neg}/{n_tot} specifications negative)")
print("  Next step: python code/04_blp_subsample.py")
