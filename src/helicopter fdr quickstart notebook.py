# Helicopter FDR Analysis - Quick Start Notebook
# Copy this structure into a Jupyter notebook for interactive analysis

# ============================================================================
# CELL 1: Import Libraries
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ All libraries imported successfully!")

# ============================================================================
# CELL 2: Load Your Data
# ============================================================================

# Replace with your actual file path
file_path = "/path/to/your/helicopter_fdr_data.csv"

# Load data
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# ============================================================================
# CELL 3: Data Exploration & Visualization
# ============================================================================

# Summary statistics
print("\n=== DATA SUMMARY ===")
print(df.describe())

# Correlation matrix (to understand relationships)
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Parameter Correlations")
plt.tight_layout()
plt.show()

# ============================================================================
# CELL 4: Data Cleaning & Preprocessing
# ============================================================================

# Select parameters to use (CUSTOMIZE THIS BASED ON YOUR DATA)
parameters_to_use = [
    'airspeed',         # Your column names here
    'altitude',
    'vertical_speed',
    'pitch',
    'roll',
    'yaw',
    'engine_rpm',
    'fuel_flow',
    'oil_temp',
    'oil_pressure',
    'vibration_x',
    'vibration_y',
    'vibration_z',
    'g_force_x',
    'g_force_y',
    'g_force_z'
]

# Filter to only available columns
available_cols = [col for col in parameters_to_use if col in df.columns]
df_selected = df[available_cols].copy()

print(f"Using {len(available_cols)} parameters")
print(f"Available: {available_cols}")

# Handle missing values
print(f"\nMissing values before cleaning: {df_selected.isnull().sum().sum()}")

# Forward fill then backward fill (for time series)
df_selected = df_selected.fillna(method='ffill').fillna(method='bfill')

# Remove rows still with NaN
df_selected = df_selected.dropna()

print(f"Missing values after cleaning: {df_selected.isnull().sum().sum()}")
print(f"Final dataset shape: {df_selected.shape}")

# ============================================================================
# CELL 5: Outlier Detection & Removal
# ============================================================================

df_clean = df_selected.copy()

# Remove extreme outliers using IQR method
for col in df_clean.columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers_before = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
    
    if outliers_before.sum() > 0:
        print(f"{col}: Removed {outliers_before.sum()} outliers")
        df_clean.loc[outliers_before, col] = np.nan

# Fill the NaN values created by outlier removal
df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

print(f"\nDataset shape after outlier removal: {df_clean.shape}")

# ============================================================================
# CELL 6: Normalize Data
# ============================================================================

# Use RobustScaler for FDR data (handles outliers better)
scaler = RobustScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=df_clean.columns,
    index=df_clean.index
)

print("Data normalized using RobustScaler")
print("\nNormalized data statistics:")
print(df_normalized.describe())

# ============================================================================
# CELL 7: Dimensionality Reduction with PCA
# ============================================================================

# Apply PCA for visualization (2D)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_normalized)

print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative Variance Explained: {np.sum(pca.explained_variance_ratio_):.2%}")

# Also compute 3-component PCA for better insight
pca_3d = PCA(n_components=3)
df_pca_3d = pca_3d.fit_transform(df_normalized)

print(f"\nWith 3 components: {np.sum(pca_3d.explained_variance_ratio_):.2%} variance explained")

# ============================================================================
# CELL 8: K-Means Clustering
# ============================================================================

# Test different numbers of clusters
inertias = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_normalized)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_normalized, kmeans.labels_))

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Curve')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score vs k')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Choose optimal k (usually where elbow happens)
optimal_k = 4  # YOU CAN ADJUST THIS
print(f"\nUsing k = {optimal_k} clusters")

# ============================================================================
# CELL 9: Fit Final K-Means Model
# ============================================================================

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(df_normalized)

print(f"Cluster distribution:")
print(pd.Series(clusters).value_counts().sort_index())
print(f"\nSilhouette Score: {silhouette_score(df_normalized, clusters):.3f}")
print(f"Davies-Bouldin Score: {davies_bouldin_score(df_normalized, clusters):.3f}")

# ============================================================================
# CELL 10: Visualize Clusters
# ============================================================================

# 2D PCA Visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, 
                      cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.scatter(kmeans_final.cluster_centers_[:, 0], 
           kmeans_final.cluster_centers_[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroids')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('K-Means Clustering (PCA 2D Visualization)')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# CELL 11: Analyze Cluster Characteristics
# ============================================================================

# Add cluster labels to original data for analysis
df_analysis = df_clean.copy()
df_analysis['cluster'] = clusters

print("\n" + "="*70)
print("CLUSTER CHARACTERISTICS (Mean Values)")
print("="*70)

for cluster_id in sorted(df_analysis['cluster'].unique()):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
    n_samples = len(cluster_data)
    pct = 100 * n_samples / len(df_analysis)
    
    print(f"\n--- CLUSTER {cluster_id} ({n_samples} samples, {pct:.1f}%) ---")
    print(cluster_data.drop('cluster', axis=1).mean())
    
    # Try to identify the cluster
    avg_speed = cluster_data.get('airspeed', pd.Series([0])).mean()
    avg_alt_rate = cluster_data.get('vertical_speed', pd.Series([0])).mean()
    
    if avg_speed < 10 and abs(avg_alt_rate) < 50:
        print("→ Likely: HOVER/STATIONARY")
    elif avg_alt_rate > 100:
        print("→ Likely: CLIMB")
    elif avg_alt_rate < -100:
        print("→ Likely: DESCENT")
    elif avg_speed > 50:
        print("→ Likely: CRUISE/FAST FLIGHT")
    else:
        print("→ Pattern: MIXED/TRANSITION")

# ============================================================================
# CELL 12: DBSCAN Anomaly Detection
# ============================================================================

dbscan = DBSCAN(eps=0.5, min_samples=10)
anomalies = dbscan.fit_predict(df_normalized)

n_clusters_dbscan = len(set(anomalies)) - (1 if -1 in anomalies else 0)
n_anomalies = list(anomalies).count(-1)

print(f"\nDBSCAN Results:")
print(f"  Normal clusters found: {n_clusters_dbscan}")
print(f"  Anomalies detected: {n_anomalies} ({100*n_anomalies/len(anomalies):.2f}%)")

# Visualize anomalies
plt.figure(figsize=(12, 8))
normal_mask = anomalies != -1
plt.scatter(df_pca[normal_mask, 0], df_pca[normal_mask, 1], 
           c='blue', s=30, alpha=0.6, label='Normal', edgecolors='black', linewidth=0.5)
plt.scatter(df_pca[~normal_mask, 0], df_pca[~normal_mask, 1], 
           c='red', marker='X', s=200, label='Anomaly', edgecolors='black', linewidth=2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('DBSCAN Anomaly Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# CELL 13: Isolation Forest - Another Anomaly Detection Method
# ============================================================================

iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_pred = iso_forest.fit_predict(df_normalized)
anomaly_scores = iso_forest.score_samples(df_normalized)

n_anomalies_if = (anomaly_pred == -1).sum()
print(f"\nIsolation Forest Results:")
print(f"  Anomalies detected: {n_anomalies_if} ({100*n_anomalies_if/len(anomaly_pred):.2f}%)")

# Visualize with anomaly scores
plt.figure(figsize=(12, 5))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=anomaly_scores, 
                      cmap='RdYlGn_r', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Anomaly Scores (Red = More Anomalous)')
plt.colorbar(scatter, label='Anomaly Score')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# CELL 14: Detailed Anomaly Analysis
# ============================================================================

# Find most anomalous samples
top_anomalies_idx = np.argsort(anomaly_scores)[:10]  # 10 most anomalous

print("\nMost Anomalous Samples:")
print(df_clean.iloc[top_anomalies_idx])

# What makes them anomalous? Compare to normal data
normal_idx = anomaly_pred == 1
anomaly_idx = anomaly_pred == -1

print("\n=== Anomaly Characteristics vs Normal ===")
comparison = pd.DataFrame({
    'Normal_Mean': df_normalized[normal_idx].mean(),
    'Anomaly_Mean': df_normalized[anomaly_idx].mean(),
    'Difference': df_normalized[anomaly_idx].mean() - df_normalized[normal_idx].mean()
})
print(comparison.sort_values('Difference', ascending=False).head(10))

# ============================================================================
# CELL 15: Time Series Anomalies (Sudden Changes)
# ============================================================================

# Detect sudden changes in parameters
window_size = 50  # 50 samples window (adjust based on your sampling rate)

anomaly_scores_ts = pd.DataFrame(index=df_clean.index)

for col in df_clean.columns:
    rolling_mean = df_clean[col].rolling(window=window_size, center=True).mean()
    rolling_std = df_clean[col].rolling(window=window_size, center=True).std()
    
    # Z-score: how many standard deviations from rolling mean
    z_score = np.abs((df_clean[col] - rolling_mean) / (rolling_std + 1e-10))
    anomaly_scores_ts[f'{col}_zscore'] = z_score

# Overall anomaly score
anomaly_scores_ts['overall'] = anomaly_scores_ts.mean(axis=1)

# Find samples with high overall anomaly score
threshold = anomaly_scores_ts['overall'].quantile(0.95)
high_anomaly_ts = anomaly_scores_ts['overall'] > threshold

print(f"\nTime Series Anomalies (sudden changes):")
print(f"Found {high_anomaly_ts.sum()} samples with sudden changes")

# Plot anomaly score over time
plt.figure(figsize=(14, 4))
plt.plot(anomaly_scores_ts['overall'].values, alpha=0.7)
plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score')
plt.title('Time Series Anomaly Score Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# CELL 16: Feature Importance (Which Parameters Matter Most?)
# ============================================================================

# Based on variance explained per cluster
feature_importance = df_analysis.groupby('cluster').var().mean()
feature_importance_sorted = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance_sorted.plot(kind='barh')
plt.xlabel('Average Variance Across Clusters')
plt.title('Feature Importance (Which Parameters Differentiate Clusters)')
plt.tight_layout()
plt.show()

# ============================================================================
# CELL 17: Summary Report
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS SUMMARY REPORT")
print("="*70)

print(f"\n1. DATASET")
print(f"   Total Samples: {len(df_clean)}")
print(f"   Parameters Used: {len(available_cols)}")
print(f"   Missing Data Handling: Forward/Backward fill")

print(f"\n2. CLUSTERING (K-MEANS)")
print(f"   Number of Clusters: {optimal_k}")
print(f"   Silhouette Score: {silhouette_score(df_normalized, clusters):.3f}")
print(f"   Davies-Bouldin Index: {davies_bouldin_score(df_normalized, clusters):.3f}")
print(f"   Cluster Distribution: {np.bincount(clusters).tolist()}")

print(f"\n3. ANOMALIES")
print(f"   DBSCAN Anomalies: {n_anomalies} ({100*n_anomalies/len(anomalies):.2f}%)")
print(f"   Isolation Forest Anomalies: {n_anomalies_if} ({100*n_anomalies_if/len(anomaly_pred):.2f}%)")
print(f"   Time Series Anomalies: {high_anomaly_ts.sum()} ({100*high_anomaly_ts.sum()/len(df_clean):.2f}%)")

print(f"\n4. INTERPRETATION")
print(f"   Most Important Features: {feature_importance_sorted.head(3).index.tolist()}")
print(f"   PCA Variance Explained (2D): {np.sum(pca.explained_variance_ratio_):.2%}")

print("\n✓ Analysis Complete!")

# ============================================================================
# CELL 18 (Optional): Save Results
# ============================================================================

# Save cluster assignments
df_results = df_clean.copy()
df_results['kmeans_cluster'] = clusters
df_results['dbscan_cluster'] = anomalies
df_results['isolation_forest_anomaly'] = anomaly_pred
df_results['anomaly_score'] = anomaly_scores

df_results.to_csv('helicopter_fdr_analysis_results.csv', index=False)
print("✓ Results saved to 'helicopter_fdr_analysis_results.csv'")
