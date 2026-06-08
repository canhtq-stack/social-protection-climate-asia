# Replication Package

**Paper**: Social Protection as Climate Adaptation: Heterogeneous Effects on Income Inequality in Asia  
**Journal**: Economics of Disasters and Climate Change, Springer  
**Author**: Tran Quang Canh, Ho Chi Minh City University of Economics and Finance (UEF)

---

## Overview

This repository contains the full replication package for the paper. The analysis
estimates heterogeneous treatment effects of social protection coverage on annual
Gini changes using Causal Forest Double Machine Learning (CausalForestDML) on a
panel of 25 Asian economies from 1990 to 2024.

---

## Repository Structure

```
├── code/
│   ├── 01_tabnet_causal.py          # XGBoost + TabNet SHAP → CausalForestDML baseline
│   ├── 02_policy_sim.py             # Break-even and cost-benefit analysis
│   ├── 03_robustness.py             # Robustness checks RC1–RC9
│   ├── 04_blp_subsample.py          # BLP moderation + institutional quartile CATEs
│   ├── 05_parametric_benchmark.py   # OLS parametric benchmark
│   ├── 06_blp_cluster_bootstrap.py  # BLP with cluster-robust SE (cross-check)
│   └── 07_calc_agric_median.py      # Agricultural GDP share statistics (WB API)
├── data/
│   └── README_data.md               # Data sources and construction instructions
├── output/
│   ├── tables/                      # CSV tables (generated)
│   ├── figures/                     # PNG figures (generated)
│   └── models/                      # Saved model objects (generated)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11
- Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `econml==0.16.0`, `pytorch-tabnet==4.1.0`, `shap==0.44.0`,
`xgboost==2.0.3`, `statsmodels==0.14.2`

---

## Data

Raw data are **not included** due to third-party licensing. See `data/README_data.md`
for download instructions from World Bank WDI, World Bank ASPIRE, Berkeley Earth,
EM-DAT, FAO STAT, and V-Dem v16.

Place the assembled panel file at `data/panel_qrei_final.csv` before running any script.

---

## Execution Order

Run scripts sequentially from the repo root:

```bash
python code/01_tabnet_causal.py          # ~45 min (CausalForestDML + TabNet SHAP)
python code/02_policy_sim.py             # ~2 min
python code/03_robustness.py             # ~6 hours (RC1 LOCO + RC2 placebo × 200)
python code/04_blp_subsample.py          # ~30 min
python code/05_parametric_benchmark.py   # ~5 min
python code/06_blp_cluster_bootstrap.py  # ~5 min
python code/07_calc_agric_median.py      # ~2 min (requires internet for WB API)
```

> **Note**: RC2 (200 placebo permutations in `03_robustness.py`) is the most
> compute-intensive step. Runtime is approximately 6–8 hours on a standard laptop
> (Intel Core i7, 32 GB RAM). Set `N_PLACEBO = 50` for a quick test run.

---

## Outputs

After running all scripts, `output/` will contain:

| File | Manuscript item |
|---|---|
| `tables/ate_summary.csv` | In-text ATE |
| `tables/cate_by_country.csv` | Table 2 / Figure 2 |
| `tables/shap_merged_table4.csv` | Table 4 |
| `tables/blp_results.csv` | Table 5 |
| `tables/subsample_rol_cate.csv` | Table 6 |
| `tables/robustness_summary.csv` | Table 7 |
| `tables/parametric_benchmark.csv` | Table 8 |
| `figures/figure3_shap_beeswarm.png` | Figure 3 |
| `figures/figureA1_robustness.png` | Figure A1 |
| `figures/figureA2_placebo.png` | Figure A2 |
| `figures/figureA3_cate_vs_tempshock.png` | Figure A3 |

---

## Reproducibility

All random seeds are fixed at `42`. Results may differ marginally across
operating systems due to floating-point differences in parallel tree building.
The key directional findings (sign and significance of country-level CATEs,
monotonic institutional gradient, placebo p < .005) are robust to these differences.

---

## License

Code is released under the MIT License.  
Data belong to their respective providers (see `data/README_data.md`).

---

## Contact

Tran Quang Canh — canhtq@uef.edu.vn  
ORCID: https://orcid.org/0000-0001-6513-9319
