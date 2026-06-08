# Data

The raw data file `panel_qrei_final.csv` is **not included** in this repository
because it is assembled from multiple third-party sources. Follow the steps below
to reconstruct it.

## Sources

| Variable group | Source | URL |
|---|---|---|
| Gini coefficient (ΔGini, gini_level) | World Bank WDI | https://databank.worldbank.org/source/world-development-indicators |
| Social protection coverage (ASPIRE) | World Bank ASPIRE | https://www.worldbank.org/en/topic/socialprotection/brief/aspire |
| Temperature anomaly | Berkeley Earth | http://berkeleyearth.org/data/ |
| Disaster count | EM-DAT | https://www.emdat.be/ |
| Rice yield | FAO STAT | https://www.fao.org/faostat/en/#data |
| Rule of law, democracy, corruption | V-Dem v16 | https://www.v-dem.net/data/the-v-dem-dataset/ |
| GDP per capita | World Bank WDI | https://databank.worldbank.org/source/world-development-indicators |

## Panel structure

- **Countries**: 25 Asian economies (ISO3 codes listed below)
- **Period**: 1990–2024 (unbalanced)
- **Unit**: country-year

### Country list

South Asia (7): IND, BGD, PAK, LKA, NPL, AFG, BTN  
SE Asia (10): VNM, PHL, IDN, THA, MYS, KHM, MMR, LAO, SGP, TLS  
East Asia (8): CHN, KOR, JPN, MNG, HKG, MAC, BRN, TWN

## Required columns

| Column | Description |
|---|---|
| `country_code` | ISO3 country code |
| `year` | Year (integer) |
| `delta_gini` | First difference of Gini coefficient (primary outcome) |
| `gini_level` | Gini coefficient level |
| `high_social_prot` | Binary: 1 if ASPIRE coverage > full-sample median (37.4%) |
| `social_prot_coverage` | ASPIRE coverage (% population) |
| `temp_shock` | Temperature anomaly z-score (detrended, 10-yr rolling) |
| `extreme_temp_shock` | Binary: 1 if \|temp_shock\| > 2 |
| `rice_yield_dev` | Rice yield deviation (z-score, detrended) |
| `log_gdp_pc` | Log GDP per capita (2015 USD, PPP) |
| `log_gdp_pc_lag1` | Lagged log GDP per capita |
| `rule_of_law` | V-Dem rule of law index [0, 1] |
| `democracy_electoral` | V-Dem electoral democracy index [0, 1] |
| `corruption_index` | V-Dem corruption absence index [0, 1] |
| `disaster_count_cy` | Count of drought/flood events (EM-DAT) |
| `gini_lag1` | Lagged Gini coefficient |
| `temp_shock_lag1` | Lagged temperature anomaly z-score |
| `gdp_pc` | GDP per capita (2015 USD, PPP) |

## Notes on variable construction

- **delta_gini**: `gini_level_t − gini_level_{t−1}` (first difference)
- **temp_shock**: country-specific OLS linear detrend, then z-score over 10-year rolling window
- **rice_yield_dev**: same detrending procedure as temp_shock
- **high_social_prot**: coded 1 if `social_prot_coverage > 37.4%` (full-sample median);
  country-years with missing ASPIRE data coded as 0 (untreated)
- **panel_qrei_final_unbalanced.csv**: same structure, unbalanced version (used in RC7)
