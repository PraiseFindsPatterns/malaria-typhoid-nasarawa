"""
Malaria and Typhoid Epidemiology Analysis: Nasarawa 2005-2022
Author: [Your Name]
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = pd.read_csv('data/malaria_typhoid_data.csv')

print("=" * 70)
print("MALARIA AND TYPHOID FEVER ANALYSIS: NASARAWA (2005-2022)")
print("=" * 70)
print("\nData Overview:")
print(df)
print(f"\nTotal observations: {len(df)} years")
print(f"Time period: {df['year'].min()} - {df['year'].max()}")

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 70)

print("\nMALARIA:")
print(f"  Mean cases: {df['malaria'].mean():.0f}")
print(f"  Median cases: {df['malaria'].median():.0f}")
print(f"  Std Dev: {df['malaria'].std():.0f}")
print(f"  Min: {df['malaria'].min()}")
print(f"  Max: {df['malaria'].max()}")

print("\nTYPHOID:")
print(f"  Mean cases: {df['typhoid'].mean():.0f}")
print(f"  Median cases: {df['typhoid'].median():.0f}")
print(f"  Std Dev: {df['typhoid'].std():.0f}")
print(f"  Min: {df['typhoid'].min()}")
print(f"  Max: {df['typhoid'].max()}")

# ============================================================================
# 2. TREND ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("2. TREND ANALYSIS (Linear Regression)")
print("=" * 70)

X = df['year'].values.reshape(-1, 1)
y_malaria = df['malaria'].values
y_typhoid = df['typhoid'].values

# Malaria trend
model_malaria = LinearRegression()
model_malaria.fit(X, y_malaria)
malaria_trend = model_malaria.predict(X)
malaria_r2 = model_malaria.score(X, y_malaria)

print(f"\nMALARIA Trend:")
print(f"  Slope: {model_malaria.coef_[0]:.2f} cases/year")
print(f"  R² (fit quality): {malaria_r2:.3f}")
if model_malaria.coef_[0] < 0:
    print(f"  Direction: DECLINING")
else:
    print(f"  Direction: INCREASING")

# Typhoid trend
model_typhoid = LinearRegression()
model_typhoid.fit(X, y_typhoid)
typhoid_trend = model_typhoid.predict(X)
typhoid_r2 = model_typhoid.score(X, y_typhoid)

print(f"\nTYPHOID Trend:")
print(f"  Slope: {model_typhoid.coef_[0]:.2f} cases/year")
print(f"  R² (fit quality): {typhoid_r2:.3f}")
if model_typhoid.coef_[0] < 0:
    print(f"  Direction: DECLINING")
else:
    print(f"  Direction: INCREASING")

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("3. CORRELATION BETWEEN MALARIA AND TYPHOID")
print("=" * 70)

correlation = df['malaria'].corr(df['typhoid'])
print(f"\nPearson Correlation Coefficient: {correlation:.3f}")
if abs(correlation) < 0.3:
    strength = "WEAK"
elif abs(correlation) < 0.7:
    strength = "MODERATE"
else:
    strength = "STRONG"
print(f"Strength: {strength}")

# ============================================================================
# 4. FORECASTING (Simple Linear Extrapolation)
# ============================================================================
print("\n" + "=" * 70)
print("4. 20-YEAR FORECAST (2023-2042)")
print("=" * 70)

future_years = np.array([2023, 2027, 2032, 2037, 2042]).reshape(-1, 1)
malaria_forecast = model_malaria.predict(future_years)
typhoid_forecast = model_typhoid.predict(future_years)

print("\nMALARIA Forecast:")
for year, cases in zip(future_years.flatten(), malaria_forecast):
    print(f"  {year}: {cases:.0f} cases")

print("\nTYPHOID Forecast:")
for year, cases in zip(future_years.flatten(), typhoid_forecast):
    print(f"  {year}: {cases:.0f} cases")

print("\n⚠️  NOTE: Forecasts assume linear trends continue. Actual cases depend")
print("    on interventions, policy changes, and external factors.")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("5. GENERATING VISUALIZATIONS...")
print("=" * 70)

# Figure 1: Time series with trend lines
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(df['year'], df['malaria'], 'o-', linewidth=2.5, markersize=6, 
        label='Malaria (Observed)', color='#d62728', alpha=0.8)
ax.plot(df['year'], malaria_trend, '--', linewidth=2, 
        label='Malaria (Trend)', color='#d62728', alpha=0.5)
ax.plot(df['year'], df['typhoid'], 's-', linewidth=2.5, markersize=6, 
        label='Typhoid (Observed)', color='#1f77b4', alpha=0.8)
ax.plot(df['year'], typhoid_trend, '--', linewidth=2, 
        label='Typhoid (Trend)', color='#1f77b4', alpha=0.5)
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
ax.set_title('Malaria and Typhoid Cases: Nasarawa General Hospital (2005-2022)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(2005, 2023, 2))
plt.tight_layout()
plt.savefig('outputs/figures/01_timeseries_with_trends.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 01_timeseries_with_trends.png")
plt.close()

# Figure 2: Forecast (20 years)
fig, ax = plt.subplots(figsize=(13, 6))
forecast_years_full = np.arange(2005, 2043)
malaria_forecast_full = model_malaria.predict(forecast_years_full.reshape(-1, 1))
typhoid_forecast_full = model_typhoid.predict(forecast_years_full.reshape(-1, 1))

# Historical data
ax.plot(df['year'], df['malaria'], 'o-', linewidth=2.5, markersize=6, 
        label='Malaria (Observed)', color='#d62728')
ax.plot(df['year'], df['typhoid'], 's-', linewidth=2.5, markersize=6, 
        label='Typhoid (Observed)', color='#1f77b4')

# Forecast (dashed lines)
ax.plot(forecast_years_full[forecast_years_full > 2022], 
        malaria_forecast_full[forecast_years_full > 2022], 
        '--', linewidth=2, label='Malaria (Forecast)', color='#d62728', alpha=0.6)
ax.plot(forecast_years_full[forecast_years_full > 2022], 
        typhoid_forecast_full[forecast_years_full > 2022], 
        '--', linewidth=2, label='Typhoid (Forecast)', color='#1f77b4', alpha=0.6)

ax.axvline(x=2022, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Data/Forecast Split')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
ax.set_xticks(range(2005, 2043, 5))
ax.set_title('Illustrative Linear Extrapolation (not a prediction):  R² = 0.12 malaria, 0.05 typhoid',
             fontsize=12, fontweight='bold')

ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/02_20year_forecast.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 02_20year_forecast.png")
plt.close()

# Figure 3: Distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df['malaria'], bins=8, color='#d62728', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Number of Cases', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Malaria Cases', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(df['typhoid'], bins=8, color='#1f77b4', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Number of Cases', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].set_title('Distribution of Typhoid Cases', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/figures/03_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 03_distribution_comparison.png")
plt.close()

# Figure 4: Year-over-year change
malaria_change = df['malaria'].diff()
typhoid_change = df['typhoid'].diff()

fig, ax = plt.subplots(figsize=(13, 6))
x = df['year'][1:]  # Skip first year (no change for it)
width = 0.35
ax.bar(x - width/2, malaria_change[1:], width, label='Malaria Change', 
       color='#d62728', alpha=0.7)
ax.bar(x + width/2, typhoid_change[1:], width, label='Typhoid Change', 
       color='#1f77b4', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Year-over-Year Change (Cases)', fontsize=12, fontweight='bold')
ax.set_title('Annual Change in Cases', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(range(2005, 2023, 2))
plt.tight_layout()
plt.savefig('outputs/figures/04_annual_change.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 04_annual_change.png")
plt.close()

# ============================================================================
# 6. EXPORT SUMMARY RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("6. EXPORTING RESULTS...")
print("=" * 70)

summary_stats = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Trend (slope)', 'R²', 'Correlation with Typhoid'],
    'Malaria': [
        f"{df['malaria'].mean():.0f}",
        f"{df['malaria'].median():.0f}",
        f"{df['malaria'].std():.0f}",
        f"{df['malaria'].min()}",
        f"{df['malaria'].max()}",
        f"{model_malaria.coef_[0]:.2f}",
        f"{malaria_r2:.3f}",
        f"{correlation:.3f}"
    ],
    'Typhoid': [
        f"{df['typhoid'].mean():.0f}",
        f"{df['typhoid'].median():.0f}",
        f"{df['typhoid'].std():.0f}",
        f"{df['typhoid'].min()}",
        f"{df['typhoid'].max()}",
        f"{model_typhoid.coef_[0]:.2f}",
        f"{typhoid_r2:.3f}",
        f"{correlation:.3f}"
    ]
})

summary_stats.to_csv('outputs/results/summary_statistics.csv', index=False)
print("  ✓ Saved: summary_statistics.csv")

# Export forecast
forecast_df = pd.DataFrame({
    'Year': forecast_years_full,
    'Malaria_Forecast': malaria_forecast_full,
    'Typhoid_Forecast': typhoid_forecast_full
})
forecast_df.to_csv('outputs/results/20year_forecast.csv', index=False)
print("  ✓ Saved: 20year_forecast.csv")

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE!")
print("=" * 70)
print("\nAll outputs saved to:")
print("  - Figures: outputs/figures/")
print("  - Data: outputs/results/")