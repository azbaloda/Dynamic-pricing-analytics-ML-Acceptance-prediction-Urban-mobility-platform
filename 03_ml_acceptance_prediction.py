import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
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
# 1. LOAD TRIPS DATA & PREPROCESS
# ==========================================
print("\n--- LOADING DATA FOR MACHINE LEARNING ---")
df = pd.read_sql_query("SELECT * FROM trips", conn)
print(f"Loaded {len(df):,} trips for Machine Learning classification.")

# Feature Engineering
df['competitor_fare_ratio'] = df['final_fare'] / (df['competitor_fare'] + 0.01)
df['fare_diff'] = df['final_fare'] - df['competitor_fare']

# Select Features for ML (Strictly excluding hidden ground-truth or future post-trip data)
feature_num = [
    'surge_multiplier', 'distance_km', 'base_fare', 'final_fare', 'competitor_fare',
    'competitor_fare_ratio', 'fare_diff', 'supply_available_drivers', 'trip_hour',
    'is_weekend', 'is_holiday'
]

feature_cat = ['weather_condition', 'vehicle_type', 'city']

# One-Hot Encoding for Categorical Features
df_encoded = pd.get_dummies(df[feature_num + feature_cat], columns=feature_cat, drop_first=True)

X = df_encoded
y = df['trip_accepted']

print(f"Features shape after Encoding: {X.shape}")
print(f"Target variable distribution (trip_accepted):")
print(y.value_counts(normalize=True) * 100)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. TRAIN & EVALUATE MODELS
# ==========================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
}

results = {}
fpr_dict, tpr_dict = {}, {}

print("\n--- MODEL TRAINING & EVALUATION ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'auc': auc
    }
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fpr_dict[name] = fpr
    tpr_dict[name] = tpr
    
    print(f"  {name} AUC-ROC Score: {auc:.4f}")
    print(f"  Classification Report for {name}:")
    print(classification_report(y_test, y_pred, digits=4))

# ==========================================
# 3. CHART 10: FEATURE IMPORTANCE
# ==========================================
print("\nGenerating Chart 10: Feature Importance...")
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feature_names = X.columns

df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(12)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_imp, x='importance', y='feature', palette='Blues_r')
plt.title('Top 12 Drivers of Trip Acceptance (Random Forest Feature Importance)', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Relative Feature Importance', fontweight='bold')
plt.ylabel('Feature', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '10_feature_importance.png'), dpi=300)
plt.close()
print("Saved: images/10_feature_importance.png")

# ==========================================
# 4. CHART 11: ROC-AUC CURVE COMPARISON
# ==========================================
print("Generating Chart 11: ROC Curve Comparison...")
plt.figure(figsize=(9, 6))

for name in models.keys():
    plt.plot(fpr_dict[name], tpr_dict[name], linewidth=2.5, label=f"{name} (AUC = {results[name]['auc']:.4f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.50)')
plt.title('ROC-AUC Curve Comparison for Trip Acceptance Classification', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '11_roc_curve.png'), dpi=300)
plt.close()
print("Saved: images/11_roc_curve.png")

# ==========================================
# 5. CHART 12: CONFUSION MATRIX
# ==========================================
print("Generating Chart 12: Confusion Matrix...")
best_model_name = max(results, key=lambda k: results[k]['auc'])
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])

plt.figure(figsize=(7, 5.5))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False,
            xticklabels=['Cancelled (0)', 'Accepted (1)'],
            yticklabels=['Cancelled (0)', 'Accepted (1)'])
plt.title(f'Confusion Matrix — Best Model: {best_model_name}', fontsize=13, pad=15, fontweight='bold')
plt.xlabel('Predicted Trip Outcome', fontweight='bold')
plt.ylabel('Actual Trip Outcome', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, '12_confusion_matrix.png'), dpi=300)
plt.close()
print("Saved: images/12_confusion_matrix.png")

conn.close()
print("\n[SUCCESS] Machine Learning Acceptance Prediction Completed!")
