import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import shap
import json
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('figures', exist_ok=True)

# 1. Load Data
df = pd.read_csv('StressLevelDataset.csv')
print("Dataset Loaded. Shape:", df.shape)

# EDA 1: Target Class Distribution
plt.figure(figsize=(6, 4))
palette = ['#2ecc71', '#f39c12', '#e74c3c']
ax = sns.countplot(x='stress_level', data=df, palette=palette, hue='stress_level', legend=False)
plt.title('Target Class Distribution (0=Low, 1=Moderate, 2=High Stress)', fontsize=12, fontweight='bold')
plt.xlabel('Stress Level')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
plt.tight_layout()
plt.savefig('figures/eda_class_distribution.png', dpi=300)
plt.close()

# EDA 2: Key Features Distribution across Stress Levels
key_features = ['anxiety_level', 'sleep_quality', 'depression', 'academic_performance']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for idx, col in enumerate(key_features):
    sns.boxplot(x='stress_level', y=col, data=df, ax=axes[idx], palette=palette, hue='stress_level', legend=False)
    axes[idx].set_title(f'Distribution of {col} by Stress Level', fontweight='bold')
    axes[idx].set_xlabel('Stress Level (0=Low, 1=Medium, 2=High)')
plt.tight_layout()
plt.savefig('figures/eda_feature_distributions.png', dpi=300)
plt.close()

# EDA 3: Correlation Heatmap
plt.figure(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, mask=mask, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Correlation Matrix of Student Stress Factors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/eda_correlation_heatmap.png', dpi=300)
plt.close()

# Split Features and Target
X = df.drop(columns=['stress_level'])
y = df['stress_level']

# Split Train/Test (80:20) stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Model Training & Multi-class Evaluation
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine (SVM)': SVC(probability=True, kernel='rbf', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric='mlogloss')
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    prec_weighted = precision_score(y_test, y_pred, average='weighted')
    rec_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision (Macro)': prec_macro,
        'Recall (Macro)': rec_macro,
        'F1-Score (Macro)': f1_macro,
        'Precision (Weighted)': prec_weighted,
        'Recall (Weighted)': rec_weighted,
        'F1-Score (Weighted)': f1_weighted
    })

results_df = pd.DataFrame(results)
print("\n--- Model Benchmark Results (3-Class) ---")
print(results_df.to_string(index=False))

# 3. Binary Classification Setup (High Stress Screening: Class 2 vs Classes 0+1)
# To evaluate Clinical/Screening Metrics: Sensitivity, Specificity, PPV, NPV & Threshold Tuning
y_train_binary = (y_train == 2).astype(int)
y_test_binary = (y_test == 2).astype(int)

binary_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(probability=True, kernel='rbf', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric='logloss')
}

binary_results = []
fitted_binary_models = {}

for name, model in binary_models.items():
    model.fit(X_train, y_train_binary)
    fitted_binary_models[name] = model
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.50).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred).ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0 # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    binary_results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision (PPV)': ppv,
        'Recall (Sensitivity)': sensitivity,
        'Specificity': specificity,
        'NPV': npv,
        'F1-Score': f1
    })

binary_results_df = pd.DataFrame(binary_results)
print("\n--- Binary Screening (High-Risk Stress) Benchmark (Threshold = 0.50) ---")
print(binary_results_df.to_string(index=False))

# 4. Probability Threshold Tuning (High-Risk Screening on Random Forest & XGBoost)
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
thresh_results = []

rf_binary = fitted_binary_models['Random Forest']
rf_proba = rf_binary.predict_proba(X_test)[:, 1]

for th in thresholds:
    pred_th = (rf_proba >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_binary, pred_th).ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0
    
    thresh_results.append({
        'Threshold': th,
        'Sensitivity (Recall)': sens,
        'Specificity': spec,
        'PPV (Precision)': ppv,
        'NPV': npv,
        'F1-Score': f1,
        'Accuracy': acc,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    })

thresh_df = pd.DataFrame(thresh_results)
print("\n--- Threshold Tuning Results (Random Forest) ---")
print(thresh_df.to_string(index=False))

# Plot Threshold Curve
plt.figure(figsize=(8, 5))
plt.plot(thresh_df['Threshold'], thresh_df['Sensitivity (Recall)'], marker='o', label='Sensitivity (Recall)', color='#e74c3c', lw=2)
plt.plot(thresh_df['Threshold'], thresh_df['Specificity'], marker='s', label='Specificity', color='#2ecc71', lw=2)
plt.plot(thresh_df['Threshold'], thresh_df['PPV (Precision)'], marker='^', label='PPV (Precision)', color='#3498db', lw=2)
plt.plot(thresh_df['Threshold'], thresh_df['NPV'], marker='d', label='NPV', color='#9b59b6', lw=2)
plt.plot(thresh_df['Threshold'], thresh_df['F1-Score'], marker='x', linestyle='--', label='F1-Score', color='#34495e', lw=2)
plt.title('Clinical Trade-off: Threshold Tuning for High-Stress Screening', fontsize=12, fontweight='bold')
plt.xlabel('Probability Threshold')
plt.ylabel('Score')
plt.axvline(x=0.20, color='gray', linestyle=':', label='Screening Cut-off (0.20)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/threshold_tuning_tradeoff.png', dpi=300)
plt.close()

# 5. SHAP Feature Importance Analysis
print("\nRunning SHAP Analysis...")
explainer = shap.TreeExplainer(rf_binary)
shap_values = explainer.shap_values(X_test)

# If shap_values is a list (for binary classification in older versions or 3D array in newer)
if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_class1 = shap_values[:, :, 1]
else:
    shap_vals_class1 = shap_values

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_class1, X_test, show=False)
plt.title('SHAP Feature Importance (Impact on Predicting High Stress)', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Save all summary stats into json for easy insertion
summary_data = {
    "dataset": {
        "instances": len(df),
        "features_count": X.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "classes": df['stress_level'].value_counts().to_dict()
    },
    "benchmark_3class": results_df.to_dict(orient='records'),
    "benchmark_binary": binary_results_df.to_dict(orient='records'),
    "threshold_tuning": thresh_df.to_dict(orient='records')
}

with open('experiment_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=4)

print("\nAll experiments and visualizations generated successfully!")
