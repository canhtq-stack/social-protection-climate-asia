"""
07_calc_agric_median.py
=======================
Fetch and compute agricultural GDP share (% of GDP) statistics
for the 25-economy panel. Used to determine agricultural dependence
classifications reported in Table 2 (manuscript).

Paper : "Social Protection as Climate Adaptation: Heterogeneous Effects
         on Income Inequality in Asia"
Journal: Economics of Disasters and Climate Change, Springer

Data source: World Bank API — indicator NV.AGR.TOTL.ZS

Outputs
-------
data/agric_gdp_share_25countries.csv
output/tables/agric_summary_statistics.csv
"""

import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd

from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "panel_qrei_final.csv"
DATA_DIR  = ROOT / "data"
TABLE_DIR = ROOT / "output" / "tables"

DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
YEAR_START    = 1990
YEAR_END      = 2024
WB_INDICATOR  = "NV.AGR.TOTL.ZS"   # Agriculture, value added (% of GDP)

# Full 25-country panel as in manuscript
PANEL_25 = [
    "IND", "BGD", "PAK", "LKA", "NPL", "AFG", "BTN",           # South Asia
    "VNM", "PHL", "IDN", "THA", "MYS", "KHM", "MMR", "LAO",    # SE Asia
    "SGP", "TLS",
    "CHN", "KOR", "JPN", "MNG", "HKG", "MAC", "BRN", "TWN",    # East Asia
]

# ── World Bank API ─────────────────────────────────────────────────────────

def fetch_wb_indicator(iso_list, indicator, y_start, y_end):
    """Fetch indicator data from World Bank API; return DataFrame."""
    iso_str  = ";".join(iso_list)
    base_url = (
        f"https://api.worldbank.org/v2/country/{iso_str}"
        f"/indicator/{indicator}"
        f"?date={y_start}:{y_end}&format=json&per_page=1000"
    )
    records = []
    page = 1

    while True:
        try:
            resp = requests.get(f"{base_url}&page={page}", timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                break

            meta  = data[0]
            items = data[1]

            for rec in items:
                if rec.get("value") is not None:
                    records.append({
                        "country_code": rec["countryiso3code"],
                        "country_name": rec["country"]["value"],
                        "year":         int(rec["date"]),
                        "agric_gdp_pct": float(rec["value"]),
                    })

            total_pages = (meta.get("total", 0) + 1000 - 1) // 1000
            print(f"   Page {page}/{total_pages} — {len(items)} records")
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.3)

        except Exception as e:
            print(f"  ERROR on page {page}: {e}")
            break

    return pd.DataFrame(records)

# ── Load panel ─────────────────────────────────────────────────────────────
print("="*70)
print("07_calc_agric_median.py — Agricultural GDP Share Statistics")
print("="*70)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

df_panel = pd.read_csv(DATA_FILE, usecols=["country_code", "year"])
df_panel["year"] = df_panel["year"].astype(int)
print(f"  Panel: {df_panel.shape}  |  Countries: {df_panel['country_code'].nunique()}")

# ── Fetch from World Bank ──────────────────────────────────────────────────
print(f"\n  Fetching {WB_INDICATOR} from World Bank API ({YEAR_START}–{YEAR_END})...")
df_wb = fetch_wb_indicator(PANEL_25, WB_INDICATOR, YEAR_START, YEAR_END)

if df_wb.empty:
    print("  ERROR: No data returned from World Bank API.")
    sys.exit(1)

print(f"  Fetched: {len(df_wb):,} records from {df_wb['country_code'].nunique()} countries")

# ── Restrict to panel country-years ───────────────────────────────────────
df = df_wb.merge(df_panel, on=["country_code", "year"], how="inner")
print(f"  Panel-restricted: {len(df):,} obs")

# ── Summary statistics ─────────────────────────────────────────────────────
print("\n--- Full-panel statistics ---")
agric = df["agric_gdp_pct"]
print(f"  Mean   : {agric.mean():.2f}%")
print(f"  Median : {agric.median():.2f}%")
print(f"  Q25    : {agric.quantile(0.25):.2f}%")
print(f"  Q75    : {agric.quantile(0.75):.2f}%")

# Country-mean approach (preferred for classification)
country_means = (
    df.groupby(["country_code", "country_name"])["agric_gdp_pct"]
    .mean()
    .reset_index()
    .rename(columns={"agric_gdp_pct": "agric_gdp_pct_mean"})
    .sort_values("agric_gdp_pct_mean", ascending=False)
)

q25_cm = country_means["agric_gdp_pct_mean"].quantile(0.25)
q50_cm = country_means["agric_gdp_pct_mean"].median()
q75_cm = country_means["agric_gdp_pct_mean"].quantile(0.75)

print(f"\n--- Country-mean statistics (used for Table 2 classifications) ---")
print(f"  Median  : {q50_cm:.2f}%  ← threshold for 'above-median' agricultural dependence")
print(f"  Q25     : {q25_cm:.2f}%  |  Q75: {q75_cm:.2f}%")

# Add quartile classification
def classify(v):
    if v > q75_cm:
        return "Very High (top quartile)"
    elif v > q50_cm:
        return "High (3rd quartile)"
    elif v > q25_cm:
        return "Moderate (2nd quartile)"
    else:
        return "Low (bottom quartile)"

country_means["agric_dependence_class"] = country_means["agric_gdp_pct_mean"].apply(classify)
country_means["agric_gdp_pct_mean"]     = country_means["agric_gdp_pct_mean"].round(2)

print("\n--- Country classifications ---")
print(country_means[["country_code", "country_name",
                      "agric_gdp_pct_mean", "agric_dependence_class"]].to_string(index=False))

# ── Save ───────────────────────────────────────────────────────────────────
out_data = DATA_DIR / "agric_gdp_share_25countries.csv"
country_means.to_csv(out_data, index=False)
print(f"\n  ✓ data/agric_gdp_share_25countries.csv")

summary_stats = pd.DataFrame([{
    "median_panel_obs":     round(agric.median(), 2),
    "median_country_mean":  round(q50_cm, 2),
    "q25_country_mean":     round(q25_cm, 2),
    "q75_country_mean":     round(q75_cm, 2),
    "n_countries":          country_means["country_code"].nunique(),
    "n_obs":                len(df),
    "note": "Country-mean median used for agricultural dependence classification in Table 2",
}])
summary_stats.to_csv(TABLE_DIR / "agric_summary_statistics.csv", index=False)
print("  ✓ output/tables/agric_summary_statistics.csv")

print(f"\nDONE — 07_calc_agric_median.py")
print(f"  Median agricultural GDP share (country-mean): {q50_cm:.1f}%  ← use in manuscript")
