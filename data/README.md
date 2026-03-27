# Data Download Instructions

Raw data files must be downloaded manually from the six sources below and saved to this directory (`data/raw/`). Raw files are excluded from version control via `.gitignore` in compliance with each provider's redistribution terms.

After downloading all files, run `python src/01_data_build.py` to merge them into the final panel dataset.

---

## File Checklist

```
data/raw/
├── wdi_gini_gdp.csv           ← World Bank WDI
├── aspire_coverage.csv        ← World Bank ASPIRE
├── berkeley_earth/            ← Berkeley Earth (one .txt file per country)
│   ├── IND_temperature.txt
│   ├── BGD_temperature.txt
│   └── ... (25 files total)
├── emdat_disasters.xlsx       ← EM-DAT
├── faostat_rice_yield.csv     ← FAO STAT
└── vdem_v16.csv               ← V-Dem v16
```

---

## Source 1 — World Bank WDI (Gini + GDP per capita)

**Variables:** `SI.POV.GINI` (Gini coefficient), `NY.GDP.PCAP.PP.KD` (GDP per capita, PPP, 2015 USD)  
**URL:** https://databank.worldbank.org/source/world-development-indicators  
**Save as:** `data/raw/wdi_gini_gdp.csv`

Download steps:
1. Go to the DataBank URL above → "World Development Indicators"
2. **Country:** select all 25 economies listed in `docs/codebook.md`
3. **Series:** `SI.POV.GINI` and `NY.GDP.PCAP.PP.KD`
4. **Time:** 1990–2024 (Annual)
5. **Download** → CSV (default World Bank format with 4 header rows is expected)

---

## Source 2 — World Bank ASPIRE (Social Protection Coverage)

**Variable:** Total social protection and labour programmes — coverage of total population (%)  
**URL:** https://www.worldbank.org/en/topic/socialprotection/brief/atlas-social-protection-resilience-equity  
**Save as:** `data/raw/aspire_coverage.csv`

Download steps:
1. Navigate to the ASPIRE data portal at the URL above
2. Select indicator: **"Coverage — Total population (%)"** (all programmes)
3. Filter by region: East Asia & Pacific and South Asia
4. Download full dataset → CSV
5. Ensure columns include: `Country Code`, `Year`, `Coverage (%)`

---

## Source 3 — Berkeley Earth (Temperature Anomaly)

**Variable:** Annual mean country-level temperature anomaly (°C)  
**URL:** https://berkeleyearth.org/data/  
**Save to:** `data/raw/berkeley_earth/` (one file per country)

Download steps:
1. Go to https://berkeleyearth.org/data/ → "Country / Regional Averages"
2. For each of the 25 countries, download the **Annual** time series (`.txt` format)
3. Rename each file as `{ISO3}_temperature.txt` using the ISO3 codes in `docs/codebook.md`
4. Place all 25 files in `data/raw/berkeley_earth/`

Expected file format (lines beginning with `%` are comments):
```
% Berkeley Earth ...
% Year  Annual  Annual_Unc
1850    -0.312   0.241
1851    -0.284   0.238
...
```

---

## Source 4 — EM-DAT (Disaster Count)

**Variable:** Declared drought and flood disaster events per country-year  
**URL:** https://www.emdat.be/ (free registration required)  
**Save as:** `data/raw/emdat_disasters.xlsx`

Download steps:
1. Register for a free account at https://www.emdat.be/
2. Go to **Advanced Search** (or Public EMDAT)
3. Filters:
   - **Disaster Type:** Flood; Drought
   - **Continent / Region:** Asia
   - **Period:** 1990–2024
4. Download → Excel (`.xlsx`)
5. Ensure columns include: `ISO`, `Year`, `Disaster Type`

---

## Source 5 — FAO STAT (Rice Yield)

**Variable:** Rice paddy yield (hg/ha)  
**URL:** https://www.fao.org/faostat/en/#data/QCL  
**Save as:** `data/raw/faostat_rice_yield.csv`

Download steps:
1. Go to FAOSTAT → **Production** → Crops and livestock products
2. Select:
   - **Elements:** Yield
   - **Items:** Rice, paddy
   - **Area:** All 25 countries (use ISO3 codes)
   - **Years:** 1990–2024
3. Download → CSV
4. Ensure columns include: `Area Code (ISO3)`, `Year`, `Value`

---

## Source 6 — V-Dem v16 (Institutional Variables)

**Variables:** `v2x_rule` (rule of law), `v2x_polyarchy` (electoral democracy), `v2x_corr` (corruption absence)  
**URL:** https://v-dem.net/data/the-v-dem-dataset/  
**Save as:** `data/raw/vdem_v16.csv`

Download steps:
1. Go to the V-Dem data download page above
2. Download **"V-Dem-CY-Full+Others-v16"** → CSV (country-year, full dataset)
3. The file is large (~200 MB); `01_data_build.py` extracts only the 4 required columns automatically:
   - `country_text_id`, `year`, `v2x_rule`, `v2x_polyarchy`, `v2x_corr`

---

## Notes

- All sources provide data under terms that permit academic use; see each provider's terms of service for redistribution restrictions.
- V-Dem data require citation: Coppedge et al. (2024). *V-Dem Codebook v16*. Varieties of Democracy (V-Dem) Project.
- EM-DAT data require citation: Guha-Sapir D. *EM-DAT: The Emergency Events Database*. UCLouvain, Brussels, Belgium. www.emdat.be.
