import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

COUNTRY_CODES = [
    # South Asia (7)
    'IND', 'BGD', 'PAK', 'LKA', 'NPL', 'AFG', 'BTN',
    # Southeast Asia (10)
    'VNM', 'PHL', 'IDN', 'THA', 'MYS', 'KHM', 'MMR', 'LAO', 'SGP', 'TLS',
    # East Asia comparators (8)
    'CHN', 'KOR', 'JPN', 'MNG', 'HKG', 'MAC', 'BRN', 'TWN',
]

YEARS = range(1990, 2025)

DATA_DIR = Path('data/raw')
OUTPUT_PATH = Path('data/processed/cleaned_panel.csv')

# Country name to ISO3 mapping (for Berkeley Earth, etc.)
# 25 Asian economies — South Asia, Southeast Asia, East Asia
COUNTRY_MAPPING = {
    # South Asia
    'India': 'IND', 'Bangladesh': 'BGD', 'Pakistan': 'PAK', 'Sri Lanka': 'LKA',
    'Nepal': 'NPL', 'Afghanistan': 'AFG', 'Bhutan': 'BTN',
    # Southeast Asia
    'Vietnam': 'VNM', 'Philippines': 'PHL', 'Indonesia': 'IDN', 'Thailand': 'THA',
    'Malaysia': 'MYS', 'Cambodia': 'KHM', 'Myanmar': 'MMR', 'Lao PDR': 'LAO',
    'Singapore': 'SGP', 'Timor-Leste': 'TLS',
    # East Asia
    'China': 'CHN', 'South Korea': 'KOR', 'Japan': 'JPN', 'Mongolia': 'MNG',
    'Hong Kong SAR': 'HKG', 'Macao SAR': 'MAC', 'Brunei Darussalam': 'BRN',
    'Taiwan': 'TWN',
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_wb_csv(filepath, value_col_name):
    
   # Load World Bank CSV (already processed by download script)

    try:
        df = pd.read_csv(filepath)
        df = df[['country_code', 'year', value_col_name]]
        df = df[
            (df['country_code'].isin(COUNTRY_CODES)) &
            (df['year'].isin(YEARS))
        ]
        return df
    except FileNotFoundError:
        print(f"  ⚠ File not found: {filepath}")
        return None

def load_vdem():
    "Load V-Dem democracy indicators"
    filepath = DATA_DIR / 'institutions/vdem_full_v14.csv'
    
    if not filepath.exists():
        print(f"  ⚠ V-Dem file not found. Creating placeholder.")
        # Return empty dataframe with correct structure
        return pd.DataFrame(columns=['country_code', 'year', 'democracy_electoral', 
                                      'democracy_liberal', 'corruption_index', 'rule_of_law'])
    
    df = pd.read_csv(filepath)
    
    # V-Dem variable names (check actual column names in downloaded file)
    vdem_vars = {
        'country_text_id': 'country_code',  # Usually 3-letter code
        'year': 'year',
        'v2x_polyarchy': 'democracy_electoral',
        'v2x_libdem': 'democracy_liberal',
        'v2x_corr': 'corruption_index',
        'v2x_rule': 'rule_of_law'
    }
    
    # Check if columns exist
    available_cols = [col for col in vdem_vars.keys() if col in df.columns]
    df = df[available_cols]
    df = df.rename(columns=vdem_vars)
    
    # Filter
    df = df[
        (df['country_code'].isin(COUNTRY_CODES)) &
        (df['year'].isin(YEARS))
    ]
    
    return df

def load_emdat():
    "Load EM-DAT disaster data"
    filepath = DATA_DIR / 'climate/emdat_disasters.csv'
    
    if not filepath.exists():
        print(f"  ⚠ EM-DAT file not found. Creating placeholder.")
        return pd.DataFrame(columns=['country_code', 'year', 'disaster_count', 
                                      'disaster_deaths', 'disaster_affected'])
    
    df = pd.read_csv(filepath)
    
    # EM-DAT column names (verify with actual file)
    # Typical structure: ISO, Year, Disaster Type, Total Deaths, Total Affected, Total Damages
    
    # Aggregate by country-year
    df_agg = df.groupby(['ISO', 'Year']).agg({
        'Disaster Type': 'count',  # Number of disasters
        'Total Deaths': lambda x: x.fillna(0).sum(),
        'Total Affected': lambda x: x.fillna(0).sum(),
        "Total Damages ('000 US$)": lambda x: x.fillna(0).sum()
    }).reset_index()
    
    df_agg = df_agg.rename(columns={
        'ISO': 'country_code',
        'Year': 'year',
        'Disaster Type': 'disaster_count',
        'Total Deaths': 'disaster_deaths',
        'Total Affected': 'disaster_affected',
        "Total Damages ('000 US$)": 'disaster_damages_k_usd'
    })
    
    df_agg = df_agg[
        (df_agg['country_code'].isin(COUNTRY_CODES)) &
        (df_agg['year'].isin(YEARS))
    ]
    
    return df_agg

def load_climate_data():
    "Load temperature and precipitation data"
    
    # Option 1: If you have NOAA CSV
    climate_file = DATA_DIR / 'climate/noaa_climate.csv'
    
    if climate_file.exists():
        df = pd.read_csv(climate_file)
        df = df[['country_code', 'year', 'temp_anomaly_celsius', 'precip_anomaly_mm']]
        return df
    
    # Option 2: If manual entry needed, use template
    else:
        print("  ⚠ Climate data not found. Using template.")
        print("  → Please fill: data/raw/climate/noaa_climate_TEMPLATE.csv")
        
        # Load template
        df = pd.read_csv(DATA_DIR / 'climate/noaa_climate_TEMPLATE.csv')
        return df

# ============================================
# MAIN PIPELINE
# ============================================

def main():
    print("="*70)
    print("QREI DATA PREPARATION PIPELINE — 25 ASIAN ECONOMIES")
    print("="*70)
    print(f"\nSample: South Asia (7) + Southeast Asia (10) + East Asia (8)")
    print(f"\nProcessing data for:")
    print(f"  - Countries: {len(COUNTRY_CODES)}")
    print(f"  - Years: {min(YEARS)}-{max(YEARS)} ({len(YEARS)} years)")
    print(f"  - Expected observations: ~{len(COUNTRY_CODES) * len(YEARS)}")
    
    # ==================== STEP 1: LOAD INEQUALITY DATA ====================
    print("\n" + "="*70)
    print("STEP 1/7: LOADING INEQUALITY DATA (BASE DATASET)")
    print("="*70)
    
    df_gini = load_wb_csv(
        DATA_DIR / 'inequality/wb_gini.csv',
        'gini'
    )
    
    if df_gini is None:
        print("  ✗ CRITICAL: Gini data not found!")
        print("  → Please run download_data.py first")
        return None
    
    print(f"  ✓ Loaded {len(df_gini)} observations")
    print(f"  ✓ Non-missing Gini: {df_gini['gini'].notna().sum()} ({df_gini['gini'].notna().sum()/len(df_gini)*100:.1f}%)")
    
    # Start with Gini as base
    df = df_gini.copy()
    
    # ==================== STEP 2: LOAD CLIMATE DATA ====================
    print("\n" + "="*70)
    print("STEP 2/7: LOADING CLIMATE DATA")
    print("="*70)
    
    # Temperature & precipitation
    df_climate = load_climate_data()
    print(f"  ✓ Loaded climate data: {len(df_climate)} observations")
    
    # Disasters
    df_disasters = load_emdat()
    print(f"  ✓ Loaded disaster data: {len(df_disasters)} observations")
    
    # Merge climate
    df = df.merge(df_climate, on=['country_code', 'year'], how='left')
    df = df.merge(df_disasters, on=['country_code', 'year'], how='left')
    
    print(f"  ✓ After merge: {len(df)} observations")
    
    # ==================== STEP 3: LOAD INSTITUTIONAL DATA ====================
    print("\n" + "="*70)
    print("STEP 3/7: LOADING INSTITUTIONAL DATA")
    print("="*70)
    
    # V-Dem
    df_vdem = load_vdem()
    if len(df_vdem) > 0:
        df = df.merge(df_vdem, on=['country_code', 'year'], how='left')
        print(f"  ✓ Merged V-Dem: {df['democracy_electoral'].notna().sum()} non-missing")
    else:
        print("  ⚠ V-Dem data empty (manual download needed)")
    
    # World Bank Governance Indicators
    wgi_vars = ['control_of_corruption', 'government_effectiveness', 'rule_of_law']
    for var in wgi_vars:
        df_wgi = load_wb_csv(DATA_DIR / f'institutions/wgi_{var}.csv', var)
        if df_wgi is not None:
            df = df.merge(df_wgi, on=['country_code', 'year'], how='left')
            print(f"  ✓ Merged WGI {var}")
    
    # ==================== STEP 4: LOAD CONTROL VARIABLES ====================
    print("\n" + "="*70)
    print("STEP 4/7: LOADING CONTROL VARIABLES")
    print("="*70)
    
    controls = {
        'gdp_pc_constant2015usd': 'controls/wb_gdp_per_capita.csv',
        'population': 'controls/wb_population.csv',
        'agriculture_pct_gdp': 'controls/wb_agriculture_pct_gdp.csv',
        'urban_population_pct': 'controls/wb_urban_pct.csv',
        'trade_pct_gdp': 'controls/wb_trade_pct_gdp.csv',
        'gdp_growth_annual_pct': 'controls/wb_gdp_growth.csv'
    }
    
    for var_name, filepath in controls.items():
        df_ctrl = load_wb_csv(DATA_DIR / filepath, var_name)
        if df_ctrl is not None:
            df = df.merge(df_ctrl, on=['country_code', 'year'], how='left')
            print(f"  ✓ Merged {var_name}")
    
    print(f"  ✓ Total variables after merge: {len(df.columns)}")
    
    # ==================== STEP 5: FEATURE ENGINEERING ====================
    print("\n" + "="*70)
    print("STEP 5/7: FEATURE ENGINEERING")
    print("="*70)
    
    # Sort by country and year
    df = df.sort_values(['country_code', 'year']).reset_index(drop=True)
    
    # Climate shock: Standardized temperature deviation
    print("  → Creating climate shock variables...")
    df['temp_shock'] = df.groupby('country_code')['temp_anomaly_celsius'].transform(
        lambda x: (x - x.rolling(10, min_periods=5).mean()) / 
                  (x.rolling(10, min_periods=5).std() + 1e-6)  # Avoid division by zero
    )
    
    # Extreme shock indicator (>2 standard deviations)
    df['extreme_temp_shock'] = (df['temp_shock'].abs() > 2).astype(float)
    
    # Drought indicator (temperature high + precipitation low)
    # (Requires precipitation data)
    if 'precip_anomaly_mm' in df.columns:
        df['drought_risk'] = (
            (df['temp_anomaly_celsius'] > df['temp_anomaly_celsius'].median()) &
            (df['precip_anomaly_mm'] < df['precip_anomaly_mm'].median())
        ).astype(float)
    
    # Log transformations
    print("  → Creating log-transformed variables...")
    df['log_gdp_pc'] = np.log(df['gdp_pc_constant2015usd'].replace(0, np.nan))
    df['log_population'] = np.log(df['population'].replace(0, np.nan))
    
    # Lagged variables (for instruments and dynamic analysis)
    print("  → Creating lagged variables...")
    for lag in [1, 3, 5]:
        df[f'gini_lag{lag}'] = df.groupby('country_code')['gini'].shift(lag)
        df[f'democracy_lag{lag}'] = df.groupby('country_code')['democracy_electoral'].shift(lag)
        df[f'gdp_pc_lag{lag}'] = df.groupby('country_code')['gdp_pc_constant2015usd'].shift(lag)
    
    # First differences (for growth analysis)
    df['gini_change'] = df.groupby('country_code')['gini'].diff()
    df['gdp_pc_change'] = df.groupby('country_code')['gdp_pc_constant2015usd'].diff()
    
    # Treatment indicators (for causal analysis)
    print("  → Creating treatment indicators...")
    if 'democracy_electoral' in df.columns:
        df['high_democracy'] = (df['democracy_electoral'] > df['democracy_electoral'].median()).astype(float)
        df['democracy_change'] = df.groupby('country_code')['democracy_electoral'].diff()
        df['democratization'] = (df['democracy_change'] > 0.1).astype(float)  # Significant increase
    
    if 'rule_of_law' in df.columns:
        df['strong_institutions'] = (df['rule_of_law'] > df['rule_of_law'].median()).astype(float)
    
    # Disaster intensity (normalized)
    if 'disaster_damages_k_usd' in df.columns:
        df['disaster_intensity'] = (
            df['disaster_damages_k_usd'] / (df['gdp_pc_constant2015usd'] * df['population'] / 1000 + 1)
        )
    
    print(f"  ✓ Created {len([c for c in df.columns if any(x in c for x in ['lag', 'shock', 'change', 'log'])])} derived features")
    
    # ==================== STEP 6: HANDLE MISSING DATA ====================
    print("\n" + "="*70)
    print("STEP 6/7: HANDLING MISSING DATA")
    print("="*70)
    
    initial_rows = len(df)
    print(f"  Initial observations: {initial_rows}")
    
    # Report missing data before cleaning
    missing_before = df[['gini', 'temp_anomaly_celsius', 'democracy_electoral', 
                          'gdp_pc_constant2015usd']].isnull().sum()
    print("\n  Missing data (key variables):")
    for var, count in missing_before.items():
        pct = count / len(df) * 100
        print(f"    {var}: {count} ({pct:.1f}%)")
    
    # Strategy 1: Drop rows with missing Gini (our outcome)
    df = df.dropna(subset=['gini'])
    print(f"\n  → After dropping missing Gini: {len(df)} rows ({initial_rows - len(df)} dropped)")
    
    # Strategy 2: Interpolate slow-moving variables
    print("\n  → Interpolating slow-moving variables...")
    slow_vars = ['democracy_electoral', 'rule_of_law', 'urban_population_pct', 
                 'control_of_corruption', 'government_effectiveness']
    
    for var in slow_vars:
        if var in df.columns:
            before = df[var].isnull().sum()
            df[var] = df.groupby('country_code')[var].transform(
                lambda x: x.interpolate(method='linear', limit=3, limit_direction='both')
            )
            after = df[var].isnull().sum()
            if before > after:
                print(f"    {var}: filled {before - after} gaps")
    
    # Strategy 3: Fill disaster variables with 0 (no disaster = no impact)
    disaster_cols = [c for c in df.columns if 'disaster' in c]
    df[disaster_cols] = df[disaster_cols].fillna(0)
    print(f"  → Filled {len(disaster_cols)} disaster variables with 0")
    
    # Strategy 4: Forward fill for very sparse data (max 1 year gap)
    df = df.sort_values(['country_code', 'year'])
    sparse_vars = ['agriculture_pct_gdp', 'trade_pct_gdp']
    for var in sparse_vars:
        if var in df.columns:
            df[var] = df.groupby('country_code')[var].fillna(method='ffill', limit=1)
    
    # Final missing data report
    print("\n  Missing data after cleaning:")
    core_vars = ['gini', 'temp_anomaly_celsius', 'democracy_electoral', 
                 'gdp_pc_constant2015usd', 'population']
    missing_after = df[core_vars].isnull().sum()
    for var, count in missing_after.items():
        pct = count / len(df) * 100
        print(f"    {var}: {count} ({pct:.1f}%)")
    
    # ==================== STEP 7: QUALITY CHECKS & SAVE ====================
    print("\n" + "="*70)
    print("STEP 7/7: QUALITY CHECKS & SAVE")
    print("="*70)
    
    # Outlier detection
    print("\n  Checking for outliers...")
    outlier_vars = ['gini', 'temp_shock', 'gdp_growth_annual_pct']
    for var in outlier_vars:
        if var in df.columns:
            q1, q3 = df[var].quantile([0.01, 0.99])
            outliers = ((df[var] < q1) | (df[var] > q3)).sum()
            print(f"    {var}: {outliers} potential outliers (outside 1-99 percentile)")
    
    # Balance check
    print("\n  Country-year balance:")
    balance = df.groupby('country_code').size()
    print(f"    Min observations per country: {balance.min()}")
    print(f"    Max observations per country: {balance.max()}")
    print(f"    Mean observations per country: {balance.mean():.1f}")
    
    # Countries with <50% data
    sparse_countries = balance[balance < len(YEARS) * 0.5].index.tolist()
    if sparse_countries:
        print(f"\n  ⚠ Countries with <50% year coverage: {', '.join(sparse_countries)}")
        print("    → Consider excluding in sensitivity analysis")
    
    # Variable summary
    print("\n" + "="*70)
    print("FINAL DATASET SUMMARY")
    print("="*70)
    print(f"\n  Observations: {len(df)}")
    print(f"  Countries: {df['country_code'].nunique()}")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Variables: {len(df.columns)}")
    print(f"\n  Coverage rates (non-missing):")
    coverage = (df.notna().sum() / len(df) * 100).sort_values(ascending=False)
    print(coverage.head(15).to_string())
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Cleaned data saved to: {OUTPUT_PATH}")
    
    # Generate descriptive statistics
    print("\n  Generating descriptive statistics...")
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    
    desc_vars = ['gini', 'temp_anomaly_celsius', 'temp_shock', 'democracy_electoral',
                 'gdp_pc_constant2015usd', 'population', 'disaster_count']
    desc_vars = [v for v in desc_vars if v in df.columns]
    
    desc_stats = df[desc_vars].describe()
    desc_stats.to_csv('results/tables/descriptive_stats.csv')
    print(f"  ✓ Saved to: results/tables/descriptive_stats.csv")
    
    # Correlation matrix
    corr_vars = ['gini', 'temp_shock', 'democracy_electoral', 'log_gdp_pc', 
                 'agriculture_pct_gdp', 'disaster_count']
    corr_vars = [v for v in corr_vars if v in df.columns and df[v].notna().sum() > 50]
    
    if len(corr_vars) > 2:
        corr_matrix = df[corr_vars].corr()
        corr_matrix.to_csv('results/tables/correlation_matrix.csv')
        print(f"  ✓ Correlation matrix saved")
    
    # Save variable definitions
    var_defs = {
        'gini': 'Gini coefficient (0-100)',
        'temp_anomaly_celsius': 'Temperature anomaly (°C relative to baseline)',
        'temp_shock': 'Standardized temperature deviation (z-score)',
        'extreme_temp_shock': 'Indicator for extreme shock (|z|>2)',
        'democracy_electoral': 'Electoral democracy index (0-1)',
        'gdp_pc_constant2015usd': 'GDP per capita (constant 2015 USD)',
        'disaster_count': 'Number of natural disasters in year',
        'log_gdp_pc': 'Log of GDP per capita',
        'gini_lag1': 'Gini coefficient lagged 1 year',
        'high_democracy': 'Indicator for above-median democracy'
    }
    
    with open('data/metadata/variable_definitions.txt', 'w') as f:
        f.write("QREI Project - Variable Definitions\n")
        f.write("="*70 + "\n\n")
        for var, definition in var_defs.items():
            f.write(f"{var}:\n  {definition}\n\n")
    
    print(f"  ✓ Variable definitions saved")
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review: results/tables/descriptive_stats.csv")
    print("  2. Check: results/tables/correlation_matrix.csv")
    print("  3. Run: python code/02_exploratory_analysis.py")
    print("="*70)
    
    return df


if __name__ == '__main__':
    df = main()
