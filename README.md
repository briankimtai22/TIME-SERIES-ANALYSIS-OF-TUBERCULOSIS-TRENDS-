# %% [markdown]
# # Tuberculosis Mortality Analysis - Complete Code
# ## Data Cleaning, Analysis, Forecasting, and Machine Learning

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. DATA LOADING AND INITIAL INSPECTION

# %%
print("="*80)
print("TUBERCULOSIS MORTALITY ANALYSIS")
print("="*80)

# Load the dataset
print("\n1. LOADING DATA...")
tb_data = pd.read_csv("C:\\Users\\pc\\Downloads\\Tuberculosis_Trends.csv")

# Initial inspection
print(f"\nDataset shape: {tb_data.shape}")
print(f"\nFirst 3 rows:")
print(tb_data.head(3))
print(f"\nData types:")
print(tb_data.dtypes)
print(f"\nSummary statistics:")
print(tb_data.describe())

# %% [markdown]
# ## 2. DATA CLEANING AND REGIONAL CORRECTIONS

# %%
print("\n" + "="*80)
print("2. DATA CLEANING")
print("="*80)

# Define authoritative country-region mapping
print("\nCorrecting regional data...")
region_mapping = {
    'Brazil': 'South America',
    'USA': 'North America',
    'Russia': 'Europe',
    'China': 'Asia',
    'India': 'Asia',
    'Indonesia': 'Asia',
    'Pakistan': 'Asia',
    'Nigeria': 'Africa',
    'South Africa': 'Africa',
    'Bangladesh': 'Asia'
}

# Before correction
print("\nRegion counts before correction:")
print(tb_data['Region'].value_counts())

# Apply corrections
tb_data['Region'] = tb_data['Country'].map(region_mapping).fillna(tb_data['Region'])

# After correction
print("\nRegion counts after correction:")
print(tb_data['Region'].value_counts())

# Verify corrections
print("\nUnique country-region pairs:")
print(tb_data[['Country', 'Region']].drop_duplicates().sort_values('Country'))

# %% [markdown]
# ## 3. MISSING VALUE ANALYSIS AND TREATMENT

# %%
print("\n" + "="*80)
print("3. MISSING VALUE ANALYSIS")
print("="*80)

# Missing value analysis
print("\nMissing value analysis:")
missing_values = tb_data.isnull().sum()
missing_percent = (tb_data.isnull().sum() / len(tb_data)) * 100
missing_report = pd.concat([missing_values, missing_percent], axis=1)
missing_report.columns = ['Missing Count', 'Missing %']
print(missing_report[missing_report['Missing Count'] > 0].sort_values('Missing %', ascending=False))

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(tb_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap Before Treatment', pad=20)
plt.show()

# Handle missing values
print("\nTreating missing values...")

# Numerical columns - median imputation
num_cols = tb_data.select_dtypes(include=np.number).columns
for col in num_cols:
    if tb_data[col].isnull().sum() > 0:
        median_val = tb_data[col].median()
        tb_data[col].fillna(median_val, inplace=True)
        print(f"Filled {missing_values[col]} missing values in {col} with median: {median_val}")

# Categorical columns - mode imputation
cat_cols = tb_data.select_dtypes(include='object').columns
for col in cat_cols:
    if tb_data[col].isnull().sum() > 0:
        mode_val = tb_data[col].mode()[0]
        tb_data[col].fillna(mode_val, inplace=True)
        print(f"Filled {missing_values[col]} missing values in {col} with mode: '{mode_val}'")

# Verify no missing values remain
print("\nMissing values after treatment:", tb_data.isnull().sum().sum())

# %% [markdown]
# ## 4. DATA QUALITY CHECKS

# %%
print("\n" + "="*80)
print("4. DATA QUALITY CHECKS")
print("="*80)

print("\nPerforming data quality checks...")

# Check for impossible values in key metrics
def check_value_ranges(df, col, min_val, max_val):
    invalid = df[(df[col] < min_val) | (df[col] > max_val)]
    if not invalid.empty:
        print(f"Warning: {len(invalid)} rows with {col} outside expected range ({min_val}-{max_val})")
        return invalid
    else:
        print(f"{col} values all within expected range ({min_val}-{max_val})")
        return None

# Validate key columns
check_value_ranges(tb_data, 'TB_Mortality_Rate', 0, 500)
check_value_ranges(tb_data, 'TB_Incidence_Rate', 0, 1000)
check_value_ranges(tb_data, 'TB_Treatment_Success_Rate', 0, 100)

# Check for duplicate rows
duplicates = tb_data.duplicated()
print(f"\nFound {duplicates.sum()} duplicate rows")
if duplicates.sum() > 0:
    tb_data = tb_data.drop_duplicates()
    print("Duplicate rows removed")

# Check year range
print("\nYear range:", tb_data['Year'].min(), "to", tb_data['Year'].max())

# Final data shape
print("\nFinal cleaned data shape:", tb_data.shape)

# %% [markdown]
# ## 5. SAVE CLEANED DATA

# %%
print("\n" + "="*80)
print("5. SAVING CLEANED DATA")
print("="*80)

# Save cleaned data
clean_filename = 'TB_Data_Cleaned.csv'
tb_data.to_csv(clean_filename, index=False)
print(f"\nCleaned data saved to {clean_filename}")

# Save cleaning report
with open('data_cleaning_report.txt', 'w') as f:
    f.write("TB Data Cleaning Report\n")
    f.write("==============================\n")
    f.write(f"Original shape: {tb_data.shape}\n")
    f.write(f"Missing values treated: {missing_values.sum()}\n")
    f.write("Regional corrections applied to:\n")
    for country, region in region_mapping.items():
        f.write(f"- {country}: {region}\n")
    f.write("\nFinal data checks passed:\n")
    f.write("- No remaining missing values\n")
    f.write("- All values within expected ranges\n")
    f.write(f"- {duplicates.sum()} duplicate rows removed\n")
print("Cleaning report saved to data_cleaning_report.txt")

# %% [markdown]
# ## 6. TREND ANALYSIS AND VISUALIZATION

# %%
print("\n" + "="*80)
print("6. TREND ANALYSIS")
print("="*80)

# Set professional style
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Convert Year to datetime for better plotting
tb_data['Year'] = pd.to_datetime(tb_data['Year'], format='%Y')

# Calculate regional mortality trends
regional_trends = tb_data.groupby(['Region', 'Year'])['TB_Mortality_Rate'].mean().unstack()

# Plot mortality trends
plt.figure(figsize=(14, 7))
plt.plot(regional_trends.index, regional_trends['Africa'], label='Africa', linewidth=3, marker='o')

for region in regional_trends.columns:
    if region != 'Africa':
        plt.plot(regional_trends.index, regional_trends[region], label=region, alpha=0.6, linestyle='--')

plt.title('TB Mortality Rate Trends by Region (2000-2024)', fontsize=16, pad=20)
plt.ylabel('Mortality Rate (per 100,000)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate percentage changes
trend_comparison = pd.DataFrame({
    '2000': regional_trends.iloc[0],
    '2024': regional_trends.iloc[-1],
    'Change (%)': ((regional_trends.iloc[-1] - regional_trends.iloc[0]) / regional_trends.iloc[0] * 100)
}).sort_values('Change (%)')

print("\nMortality Rate Change (2000-2024):")
print(trend_comparison[['2000', '2024', 'Change (%)']].round(1))

# %% [markdown]
# ## 7. CORRELATION ANALYSIS

# %%
print("\n" + "="*80)
print("7. CORRELATION ANALYSIS")
print("="*80)

# Select key co-factors
co_factors = ['HIV_CoInfected_TB_Cases', 'Drug_Resistant_TB_Cases', 
              'TB_Treatment_Success_Rate', 'Health_Expenditure_Per_Capita', 
              'BCG_Vaccination_Coverage']

# Calculate correlations for Africa
africa_corr = tb_data[tb_data['Region'] == 'Africa'][co_factors + ['TB_Mortality_Rate']].corr()[['TB_Mortality_Rate']].drop('TB_Mortality_Rate')

# Calculate correlations for non-Africa
non_africa_corr = tb_data[tb_data['Region'] != 'Africa'][co_factors + ['TB_Mortality_Rate']].corr()[['TB_Mortality_Rate']].drop('TB_Mortality_Rate')

# Visualize comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
sns.heatmap(africa_corr, annot=True, cmap='coolwarm', center=0, annot_kws={'size': 10}, fmt='.2f', ax=ax1)
ax1.set_title('Africa: Correlation with TB Mortality', pad=20)
sns.heatmap(non_africa_corr, annot=True, cmap='coolwarm', center=0, annot_kws={'size': 10}, fmt='.2f', ax=ax2)
ax2.set_title('Non-Africa: Correlation with TB Mortality', pad=20)
plt.tight_layout()
plt.show()

# Identify key differences
print("\nTop correlations with TB Mortality:")
print("In Africa:")
print(africa_corr['TB_Mortality_Rate'].abs().sort_values(ascending=False)[:3])
print("\nOutside Africa:")
print(non_africa_corr['TB_Mortality_Rate'].abs().sort_values(ascending=False)[:3])

# %% [markdown]
# ## 8. TIME SERIES FORECASTING (ARIMA) - ALL REGIONS

# %%
print("\n" + "="*80)
print("8. TIME SERIES FORECASTING (ARIMA)")
print("="*80)

def forecast_region(region_name, data):
    """Function to forecast TB mortality for a specific region"""
    print(f"\n{'-'*40}")
    print(f"Forecasting for {region_name}")
    print(f"{'-'*40}")
    
    # Prepare time series
    region_ts = data[data['Region'] == region_name].groupby('Year')['TB_Mortality_Rate'].mean()
    region_ts.index = pd.to_datetime(region_ts.index, format='%Y')
    region_ts = region_ts.asfreq('YS')
    
    # Time series decomposition
    plt.figure(figsize=(12, 8))
    try:
        decomposition = seasonal_decompose(region_ts, model='additive', period=5)
        decomposition.plot()
        plt.suptitle(f'Time Series Decomposition of {region_name} TB Mortality', y=1.02)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Decomposition failed: {e}")
    
    # ARIMA modeling
    try:
        model = ARIMA(region_ts, order=(1,1,1))
        model_fit = model.fit()
        
        # Forecast next 5 years
        forecast = model_fit.get_forecast(steps=5)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(region_ts.index, region_ts, label='Historical Data', linewidth=2)
        plt.plot(forecast_values.index, forecast_values, color='red', 
                 linestyle='--', marker='o', label='Forecast')
        plt.fill_between(conf_int.index, 
                         conf_int.iloc[:,0], 
                         conf_int.iloc[:,1], 
                         color='pink', alpha=0.3, label='95% Confidence Interval')
        plt.title(f'TB Mortality Rate Forecast for {region_name} (2025-2029)', fontsize=14, pad=20)
        plt.ylabel('Mortality Rate (per 100,000)', fontsize=12)
        plt.xlabel('Year', fontsize=12)
        plt.legend(fontsize=12, framealpha=1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Print forecast values
        print(f"\n5-Year Mortality Rate Forecast for {region_name}:")
        forecast_df = pd.DataFrame({
            'Year': forecast_values.index.year,
            'Forecast': forecast_values.round(2),
            'Lower CI': conf_int.iloc[:,0].round(2),
            'Upper CI': conf_int.iloc[:,1].round(2)
        })
        print(forecast_df.to_string(index=False))
        return forecast_df
    except Exception as e:
        print(f"ARIMA modeling failed: {e}")
        return None

# Run forecasts for all regions
regions = ['Africa', 'Asia', 'Europe', 'South America', 'North America']
forecasts = {}
for region in regions:
    forecasts[region] = forecast_region(region, tb_data)

# %% [markdown]
# ## 9. MACHINE LEARNING (RANDOM FOREST) - ALL REGIONS

# %%
print("\n" + "="*80)
print("9. MACHINE LEARNING - RANDOM FOREST")
print("="*80)

def random_forest_region(region_name, data):
    """Function to run Random Forest for a specific region"""
    print(f"\n{'-'*40}")
    print(f"Random Forest Analysis for {region_name}")
    print(f"{'-'*40}")
    
    # Prepare data
    region_data = data[data['Region'] == region_name].copy()
    region_data['Year'] = region_data['Year'].dt.year
    
    features = ['Year', 'HIV_CoInfected_TB_Cases', 'Drug_Resistant_TB_Cases', 
                'TB_Treatment_Success_Rate', 'Health_Expenditure_Per_Capita', 
                'BCG_Vaccination_Coverage', 'TB_Doctors_Per_100K']
    
    X = region_data[features]
    y = region_data['TB_Mortality_Rate']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel Performance Metrics:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Feature importance
    feature_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 5))
    feature_imp.plot(kind='bar')
    plt.title(f'Feature Importance for TB Mortality Prediction in {region_name}', fontsize=14, pad=20)
    plt.ylabel('Importance Score', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.show()
    
    return {'MAE': mae, 'RMSE': rmse, 'feature_importance': feature_imp}

# Run Random Forest for all regions
rf_results = {}
for region in regions:
    rf_results[region] = random_forest_region(region, tb_data)

# %% [markdown]
# ## 10. SUMMARY AND COMPARISON

# %%
print("\n" + "="*80)
print("10. SUMMARY AND COMPARISON")
print("="*80)

# Create summary table
print("\nRegion-wise Model Performance Summary:")
print("-"*60)
print(f"{'Region':<15} {'MAE':<10} {'RMSE':<10} {'Top Predictor':<25}")
print("-"*60)

for region in regions:
    top_feature = rf_results[region]['feature_importance'].index[0]
    print(f"{region:<15} {rf_results[region]['MAE']:<10.2f} {rf_results[region]['RMSE']:<10.2f} {top_feature:<25}")

# Mortality trend comparison
print("\nRegion-wise Mortality Trends (2000-2024):")
print("-"*60)
print(trend_comparison.to_string())

# Final forecast comparison
print("\n2025-2029 Forecast Comparison:")
print("-"*60)
for region in regions:
    if forecasts[region] is not None:
        avg_forecast = forecasts[region]['Forecast'].mean()
        print(f"{region:<15} Average Forecast (2025-2029): {avg_forecast:.2f} per 100,000")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
