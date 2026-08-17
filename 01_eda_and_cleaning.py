import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)

db_path = 'data/pricing_analytics.db'
print("Connecting to database:", db_path)
conn = sqlite3.connect(db_path)

# ==========================================
# 1. LOAD DATA & DATA CLEANING
# ==========================================
print("\n--- 1. LOADING DATASETS ---")
df_trips = pd.read_sql_query("SELECT * FROM trips", conn)
df_customers = pd.read_sql_query("SELECT * FROM customers", conn)
df_drivers = pd.read_sql_query("SELECT * FROM drivers", conn)
df_events = pd.read_sql_query("SELECT * FROM surge_events", conn)

print(f"Loaded Trips: {df_trips.shape[0]:,} rows, {df_trips.shape[1]} columns")
print(f"Loaded Customers: {df_customers.shape[0]:,} rows")
print(f"Loaded Drivers: {df_drivers.shape[0]:,} rows")
print(f"Loaded Surge Events: {df_events.shape[0]:,} rows")

# Merge customer price sensitivity for EDA
df_trips = df_trips.merge(
    df_customers[['customer_id', 'price_sensitivity', 'customer_segment']],
    on='customer_id',
    how='left'
)

# Feature Engineering
df_trips['competitor_fare_diff'] = df_trips['final_fare'] - df_trips['competitor_fare']
df_trips['competitor_fare_ratio'] = df_trips['final_fare'] / (df_trips['competitor_fare'] + 0.01)

# Surge Multiplier Buckets
bins = [0.99, 1.0, 1.4, 1.9, 2.4, 3.0, 5.0]
labels = ['1.0x (Base)', '1.1x-1.4x (Low)', '1.5x-1.9x (Mod)', '2.0x-2.4x (High)', '2.5x-3.0x (Very High)', '>3.0x (Extreme)']
df_trips['surge_bucket'] = pd.cut(df_trips['surge_multiplier'], bins=bins, labels=labels)

print("\n--- DATA CLEANING & CHECKS ---")
print("Null values count in Trips:")
print(df_trips.isnull().sum()[df_trips.isnull().sum() > 0])

# Summary Statistics
print("\n--- REVENUE & METRICS SUMMARY ---")
completed_trips = df_trips[df_trips['trip_accepted'] == 1]
total_gross_revenue = completed_trips['final_fare'].sum()
total_requested_revenue = df_trips['final_fare'].sum()
overall_acceptance_rate = (df_trips['trip_accepted'].mean()) * 100

print(f"Total Trip Requests: {len(df_trips):,}")
print(f"Completed Trips: {len(completed_trips):,} ({overall_acceptance_rate:.2f}% Conversion)")
print(f"Total Requested Value: ₹{total_requested_revenue/1e7:.2f} Cr")
print(f"Total Completed Gross Revenue: ₹{total_gross_revenue/1e7:.2f} Cr")
print(f"Revenue Lost to Cancellations: ₹{(total_requested_revenue - total_gross_revenue)/1e7:.2f} Cr")

# ==========================================
# 2. EDA CHART 1: CITY-WISE PERFORMANCE
# ==========================================
print("\nGenerating Chart 1: City Performance...")
city_perf = df_trips.groupby('city').agg(
    total_requests=('trip_id', 'count'),
    accepted_trips=('trip_accepted', 'sum'),
    conversion_rate=('trip_accepted', lambda x: x.mean() * 100),
    total_revenue=('final_fare', lambda x: x[df_trips.loc[x.index, 'trip_accepted'] == 1].sum() / 1e5), # ₹ Lakhs
    avg_surge=('surge_multiplier', 'mean')
).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

sns.barplot(data=city_perf, x='city', y='total_revenue', color='#2b5c8f', ax=ax1, alpha=0.85)
sns.lineplot(data=city_perf, x='city', y='conversion_rate', color='#e74c3c', marker='o', linewidth=2.5, markersize=8, ax=ax2)

ax1.set_title('Total Completed Revenue (₹ Lakhs) & Acceptance Rate (%) by City', fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('Metro City', fontweight='bold')
ax1.set_ylabel('Completed Revenue (₹ Lakhs)', color='#2b5c8f', fontweight='bold')
ax2.set_ylabel('Trip Acceptance Rate (%)', color='#e74c3c', fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(False)

for i, row in city_perf.iterrows():
    ax1.text(i, row['total_revenue'] + 50, f"₹{row['total_revenue']:.0f}L", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.text(i, row['conversion_rate'] + 2, f"{row['conversion_rate']:.1f}%", ha='center', va='bottom', color='#c0392b', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '01_city_revenue_performance.png'), dpi=300)
plt.close()
print("Saved: images/01_city_revenue_performance.png")

# ==========================================
# 3. EDA CHART 2: HOURLY DEMAND & SURGE CURVE
# ==========================================
print("Generating Chart 2: Hourly Demand & Surge Curve...")
hourly_perf = df_trips.groupby('trip_hour').agg(
    request_count=('trip_id', 'count'),
    avg_surge=('surge_multiplier', 'mean'),
    acceptance_rate=('trip_accepted', lambda x: x.mean() * 100)
).reset_index()

fig, ax1 = plt.subplots(figsize=(13, 6))
ax2 = ax1.twinx()

ax1.plot(hourly_perf['trip_hour'], hourly_perf['request_count'], color='#16a085', marker='s', linewidth=2.5, label='Trip Requests')
ax2.plot(hourly_perf['trip_hour'], hourly_perf['avg_surge'], color='#d35400', marker='o', linewidth=2.5, linestyle='--', label='Avg Surge Multiplier')

ax1.set_title('Hourly Trip Demand Volume vs Average Surge Multiplier Multiplier Curve', fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('Hour of Day (0-23)', fontweight='bold')
ax1.set_ylabel('Trip Request Volume', color='#16a085', fontweight='bold')
ax2.set_ylabel('Average Surge Multiplier (x)', color='#d35400', fontweight='bold')
ax1.set_xticks(range(0, 24))
ax2.grid(False)

# Highlight Peak Morning & Evening Surge Hours
ax1.axvspan(8, 10, color='#f39c12', alpha=0.2, label='Morning Rush (8-10 AM)')
ax1.axvspan(17, 20, color='#e67e22', alpha=0.2, label='Evening Rush (5-8 PM)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '02_hourly_surge_demand_curve.png'), dpi=300)
plt.close()
print("Saved: images/02_hourly_surge_demand_curve.png")

# ==========================================
# 4. EDA CHART 3: WEATHER IMPACT ON SURGE
# ==========================================
print("Generating Chart 3: Weather Impact Analysis...")
weather_perf = df_trips.groupby('weather_condition').agg(
    request_count=('trip_id', 'count'),
    avg_surge=('surge_multiplier', 'mean'),
    acceptance_rate=('trip_accepted', lambda x: x.mean() * 100),
    avg_available_drivers=('supply_available_drivers', 'mean')
).reindex(['CLEAR', 'LIGHT_RAIN', 'HEAVY_RAIN', 'FOG']).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=weather_perf, x='weather_condition', y='avg_surge', palette='YlOrRd', ax=axes[0])
axes[0].set_title('Average Surge Multiplier by Weather Condition', fontweight='bold')
axes[0].set_ylabel('Average Surge (x)')
axes[0].set_xlabel('Weather Condition')
for i, row in weather_perf.iterrows():
    axes[0].text(i, row['avg_surge'] + 0.05, f"{row['avg_surge']:.2f}x", ha='center', fontweight='bold')

sns.barplot(data=weather_perf, x='weather_condition', y='acceptance_rate', palette='Blues_r', ax=axes[1])
axes[1].set_title('Trip Acceptance Rate (%) by Weather Condition', fontweight='bold')
axes[1].set_ylabel('Acceptance Rate (%)')
axes[1].set_xlabel('Weather Condition')
axes[1].set_ylim(0, 100)
for i, row in weather_perf.iterrows():
    axes[1].text(i, row['acceptance_rate'] + 1.5, f"{row['acceptance_rate']:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '03_weather_surge_impact.png'), dpi=300)
plt.close()
print("Saved: images/03_weather_surge_impact.png")

# ==========================================
# 5. EDA CHART 4: DEMAND-SUPPLY HEATMAP
# ==========================================
print("Generating Chart 4: Demand-Supply Heatmap...")
pivot_supply = df_trips.pivot_table(
    index='city',
    columns='trip_hour',
    values='supply_available_drivers',
    aggfunc='mean'
)

plt.figure(figsize=(14, 6))
sns.heatmap(pivot_supply, cmap='RdYlGn', annot=True, fmt='.0f', cbar_kws={'label': 'Avg Available Drivers'}, linewidths=0.5)
plt.title('Average Driver Supply Availability Heatmap (City vs Hour of Day)', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Hour of Day (0-23)', fontweight='bold')
plt.ylabel('Metro City', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '04_demand_supply_heatmap.png'), dpi=300)
plt.close()
print("Saved: images/04_demand_supply_heatmap.png")

# ==========================================
# 6. EDA CHART 5: VEHICLE TYPE DISTRIBUTION
# ==========================================
print("Generating Chart 5: Vehicle Type Performance...")
vehicle_perf = df_trips.groupby('vehicle_type').agg(
    request_count=('trip_id', 'count'),
    avg_base_fare=('base_fare', 'mean'),
    avg_final_fare=('final_fare', 'mean'),
    acceptance_rate=('trip_accepted', lambda x: x.mean() * 100)
).reindex(['BIKE', 'AUTO', 'MINI', 'SEDAN', 'SUV']).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(data=vehicle_perf, x='vehicle_type', y='request_count', palette='viridis', ax=ax1, alpha=0.85)
ax1.set_title('Trip Request Share & Average Fare by Vehicle Type', fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('Vehicle Category', fontweight='bold')
ax1.set_ylabel('Total Trip Requests', fontweight='bold')

for i, row in vehicle_perf.iterrows():
    ax1.text(i, row['request_count'] + 2000, f"{row['request_count']:,}\n(Avg: ₹{row['avg_final_fare']:.0f})", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '05_vehicle_type_performance.png'), dpi=300)
plt.close()
print("Saved: images/05_vehicle_type_performance.png")

conn.close()
print("\nPhase 1 EDA & Cleaning Script Execution Finished!")
