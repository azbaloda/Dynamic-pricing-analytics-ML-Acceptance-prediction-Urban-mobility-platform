import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
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
# 1. LOAD TRIPS DATA
# ==========================================
print("\n--- LOADING TRIPS DATA ---")
df_trips = pd.read_sql_query("SELECT * FROM trips", conn)
print(f"Loaded {len(df_trips):,} trips for Advanced Analytics.")

# Engineer competitor fare ratio
df_trips['competitor_fare_ratio'] = df_trips['final_fare'] / (df_trips['competitor_fare'] + 0.01)

# Surge Multiplier Buckets
bins = [0.99, 1.0, 1.4, 1.9, 2.4, 3.0, 5.0]
labels = ['1.0x (Base)', '1.1x-1.4x (Low)', '1.5x-1.9x (Mod)', '2.0x-2.4x (High)', '2.5x-3.0x (Very High)', '>3.0x (Extreme)']
df_trips['surge_bucket'] = pd.cut(df_trips['surge_multiplier'], bins=bins, labels=labels)

# ==========================================
# 2. PROBLEM STATEMENT 2: SURGE CANCELLATION FUNNEL
# ==========================================
print("\n--- PROBLEM STATEMENT 2: SURGE CANCELLATION FUNNEL ---")
funnel_summary = df_trips.groupby('surge_bucket', observed=False).agg(
    total_requests=('trip_id', 'count'),
    accepted_trips=('trip_accepted', 'sum'),
    conversion_rate=('trip_accepted', lambda x: x.mean() * 100),
    cancellation_rate=('trip_accepted', lambda x: (1 - x.mean()) * 100),
    total_requested_val=('final_fare', 'sum'),
    lost_revenue=('final_fare', lambda x: x[df_trips.loc[x.index, 'trip_accepted'] == 0].sum() / 1e5) # ₹ Lakhs
).reset_index()

print(funnel_summary.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

sns.barplot(data=funnel_summary, x='surge_bucket', y='total_requests', color='#34495e', alpha=0.8, ax=ax1)
sns.lineplot(data=funnel_summary, x='surge_bucket', y='conversion_rate', color='#e74c3c', marker='o', linewidth=3, markersize=9, ax=ax2)

ax1.set_title('Trip Request Volume & Conversion Drop-Off Across Surge Multipliers', fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('Surge Multiplier Bucket', fontweight='bold')
ax1.set_ylabel('Total Trip Requests', color='#34495e', fontweight='bold')
ax2.set_ylabel('Conversion / Acceptance Rate (%)', color='#e74c3c', fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(False)

for i, row in funnel_summary.iterrows():
    ax2.text(i, row['conversion_rate'] + 2, f"{row['conversion_rate']:.1f}%", ha='center', color='#c0392b', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '06_cancellation_funnel_by_surge.png'), dpi=300)
plt.close()
print("Saved: images/06_cancellation_funnel_by_surge.png")

# ==========================================
# 3. PROBLEM STATEMENT 3: PRICE ELASTICITY OF DEMAND (PED)
# ==========================================
print("\n--- PROBLEM STATEMENT 3: PRICE ELASTICITY OF DEMAND (PED) ---")

ped_list = []
vehicle_types = df_trips['vehicle_type'].unique()

for v in vehicle_types:
    df_v = df_trips[df_trips['vehicle_type'] == v]
    
    # Base acceptance rate at 1.0x surge
    base_acc = df_v[df_v['surge_multiplier'] == 1.0]['trip_accepted'].mean()
    base_surge = 1.0
    
    surge_steps = [1.2, 1.5, 2.0, 2.5, 3.0]
    for s in surge_steps:
        surge_acc = df_v[df_v['surge_multiplier'] == s]['trip_accepted'].mean()
        if pd.notna(surge_acc) and base_acc > 0:
            pct_change_demand = (surge_acc - base_acc) / base_acc
            pct_change_price = (s - base_surge) / base_surge
            ped = pct_change_demand / pct_change_price
            ped_list.append({'vehicle_type': v, 'surge_multiplier': s, 'acceptance_rate': surge_acc * 100, 'ped': round(ped, 2)})

df_ped = pd.DataFrame(ped_list)
print("\nSample Price Elasticity Table:")
print(df_ped.head(10).to_string(index=False))

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_ped, x='surge_multiplier', y='ped', hue='vehicle_type', marker='o', linewidth=2.5)
plt.axhline(-1.0, color='black', linestyle='--', label='Unit Elasticity (PED = -1.0)')
plt.title('Price Elasticity of Demand (PED) Curve by Vehicle Type', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Surge Multiplier (x)', fontweight='bold')
plt.ylabel('Price Elasticity Coefficient (PED)', fontweight='bold')
plt.legend(title='Vehicle Type')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '07_price_elasticity_curve.png'), dpi=300)
plt.close()
print("Saved: images/07_price_elasticity_curve.png")

# ==========================================
# 4. PROBLEM STATEMENT 4: REVENUE OPTIMIZATION CURVE
# ==========================================
print("\n--- PROBLEM STATEMENT 4: OPTIMAL SURGE MULTIPLIER DISCOVERY ---")

rev_opt_list = []
surge_levels = np.arange(1.0, 4.1, 0.1)

for v in vehicle_types:
    df_v = df_trips[df_trips['vehicle_type'] == v]
    avg_base_fare = df_v['base_fare'].mean()
    
    for s in surge_levels:
        s_rounded = round(s, 1)
        acc_rate = df_v[df_v['surge_multiplier'] == s_rounded]['trip_accepted'].mean()
        if pd.isna(acc_rate):
            acc_rate = 0.0
            
        # Expected Revenue per request = Base Fare * Surge * Acceptance Probability
        expected_revenue = avg_base_fare * s_rounded * acc_rate
        rev_opt_list.append({'vehicle_type': v, 'surge_multiplier': s_rounded, 'expected_revenue': expected_revenue, 'acc_rate': acc_rate})

df_rev_opt = pd.DataFrame(rev_opt_list)

# Find optimal surge per vehicle
optimal_surge_summary = df_rev_opt.loc[df_rev_opt.groupby('vehicle_type')['expected_revenue'].idxmax()]
print("\nOptimal Revenue-Maximizing Surge Multiplier per Vehicle Type:")
print(optimal_surge_summary[['vehicle_type', 'surge_multiplier', 'expected_revenue', 'acc_rate']].to_string(index=False))

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_rev_opt, x='surge_multiplier', y='expected_revenue', hue='vehicle_type', linewidth=2.5)
plt.title('Expected Revenue Curve & Optimal Surge Discovery ($s^*$)', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Surge Multiplier (x)', fontweight='bold')
plt.ylabel('Expected Revenue per Request (₹)', fontweight='bold')
plt.legend(title='Vehicle Type')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '08_optimal_surge_revenue_curve.png'), dpi=300)
plt.close()
print("Saved: images/08_optimal_surge_revenue_curve.png")

# ==========================================
# 5. PROBLEM STATEMENT 6: K-MEANS CUSTOMER PRICE SENSITIVITY INFERENCE
# ==========================================
print("\n--- PROBLEM STATEMENT 6: K-MEANS BEHAVIORAL SENSITIVITY CLUSTERING ---")
print("Extracting behavioral telemetry features from trips WITHOUT ground-truth column...")

# Engineer Customer Behavioral Features purely from Trips telemetry
df_cust_features = df_trips.groupby('customer_id').agg(
    total_requests=('trip_id', 'count'),
    cancellation_rate=('trip_accepted', lambda x: (1 - x.mean())),
    surge_cancellation_rate=('trip_accepted', lambda x: (1 - x[df_trips.loc[x.index, 'surge_multiplier'] > 1.4].mean()) if any(df_trips.loc[x.index, 'surge_multiplier'] > 1.4) else 0.0),
    avg_surge_paid=('surge_multiplier', lambda x: x[df_trips.loc[x.index, 'trip_accepted'] == 1].mean() if any(df_trips.loc[x.index, 'trip_accepted'] == 1) else 1.0),
    competitor_ratio_avg=('competitor_fare_ratio', 'mean'),
    budget_vehicle_pct=('vehicle_type', lambda x: (x.isin(['BIKE', 'AUTO'])).mean()),
    premium_vehicle_pct=('vehicle_type', lambda x: (x.isin(['SEDAN', 'SUV'])).mean())
).reset_index()

# Fill any NaN in aggregated features
df_cust_features = df_cust_features.fillna(0.0)

feature_cols = ['total_requests', 'surge_cancellation_rate', 'avg_surge_paid', 'competitor_ratio_avg', 'budget_vehicle_pct', 'premium_vehicle_pct']

X = df_cust_features[feature_cols]

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means Clustering (n_clusters = 3)
print("Fitting K-Means Model (n_clusters = 3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cust_features['cluster'] = kmeans.fit_predict(X_scaled)

# Calculate Silhouette Score
sil_score = silhouette_score(X_scaled, df_cust_features['cluster'], sample_size=10000, random_state=42)
print(f"K-Means Silhouette Score: {sil_score:.4f}")

# Profile Clusters to assign personas
cluster_profile = df_cust_features.groupby('cluster')[feature_cols].mean().reset_index()
print("\nCluster Centroids Profile:")
print(cluster_profile.to_string(index=False))

# Map Cluster IDs to Business Personas based on surge_cancellation_rate & budget_vehicle_pct
# Highest surge cancellation rate -> High Price Sensitivity
sorted_clusters = cluster_profile.sort_values(by='surge_cancellation_rate', ascending=False)['cluster'].tolist()
persona_map = {
    sorted_clusters[0]: 'HIGH (Budget Sensitive)',
    sorted_clusters[1]: 'MEDIUM (Regular Commuter)',
    sorted_clusters[2]: 'LOW (Inelastic / Corporate)'
}

df_cust_features['inferred_sensitivity'] = df_cust_features['cluster'].map(persona_map)

# Merge with Ground Truth to validate ML discovery accuracy!
df_customers_gt = pd.read_sql_query("SELECT customer_id, price_sensitivity as ground_truth_sensitivity FROM customers", conn)
validation_df = df_cust_features.merge(df_customers_gt, on='customer_id')

print("\n--- ML DISCOVERY VS GROUND TRUTH VALIDATION MATRIX ---")
ct = pd.crosstab(validation_df['inferred_sensitivity'], validation_df['ground_truth_sensitivity'], margins=True)
print(ct)

# Calculate Adjusted Rand Index (ARI) score
ari_score = adjusted_rand_score(validation_df['ground_truth_sensitivity'], validation_df['inferred_sensitivity'])
print(f"\nAdjusted Rand Index (ARI Score vs Ground Truth): {ari_score:.4f}")

# Plot K-Means Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cust_features,
    x='surge_cancellation_rate',
    y='avg_surge_paid',
    hue='inferred_sensitivity',
    palette={'HIGH (Budget Sensitive)': '#e74c3c', 'MEDIUM (Regular Commuter)': '#f39c12', 'LOW (Inelastic / Corporate)': '#2ecc71'},
    alpha=0.6,
    s=30
)
plt.title('K-Means Inferred Customer Price Sensitivity Clusters', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Surge Cancellation Rate (When Surge > 1.4x)', fontweight='bold')
plt.ylabel('Average Surge Multiplier Paid (x)', fontweight='bold')
plt.legend(title='Inferred Persona')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '09_kmeans_customer_segments.png'), dpi=300)
plt.close()
print("Saved: images/09_kmeans_customer_segments.png")

conn.close()
print("\n[SUCCESS] Advanced Analytics & K-Means Script Execution Completed!")
