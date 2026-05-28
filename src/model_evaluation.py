"""
MODEL EVALUATION SCRIPT
Check how good your clustering model is
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ============================================================================
# Load and prepare data (same as your quickstart)
# ============================================================================

file_path = "../sample_fdr_data.jsonld"

with open(file_path, 'r') as f:
    fdr_data = json.load(f)

flight_data_points = fdr_data['flightData']

data_rows = []
for point in flight_data_points:
    row = {}
    for key, value in point.items():
        if isinstance(value, dict) and 'value' in value:
            row[key] = value['value']
        elif key not in ['@type']:
            row[key] = value
    data_rows.append(row)

df = pd.DataFrame(data_rows)
df_numeric = df.select_dtypes(include=[np.number])
df_clean = df_numeric.copy()

# Normalize
scaler = RobustScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=df_clean.columns
)

print("="*70)
print("MODEL EVALUATION REPORT")
print("="*70)

# ============================================================================
# 1. CLUSTERING EVALUATION METRICS
# ============================================================================

print("\n[1] CLUSTERING QUALITY METRICS")
print("-" * 70)

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_normalized)

# Silhouette Score (closer to 1 is better, -1 is worst)
sil_score = silhouette_score(df_normalized, clusters)
print(f"Silhouette Score: {sil_score:.3f}")
print(f"  → Interpretation: ", end="")
if sil_score > 0.5:
    print("✓ STRONG clusters (well-separated)")
elif sil_score > 0.3:
    print("✓ GOOD clusters (acceptable separation)")
elif sil_score > 0:
    print("⚠ WEAK clusters (overlapping)")
else:
    print("✗ BAD clusters (incorrect structure)")

# Davies-Bouldin Index (lower is better, 0 is perfect)
db_score = davies_bouldin_score(df_normalized, clusters)
print(f"\nDavies-Bouldin Index: {db_score:.3f}")
print(f"  → Interpretation: ", end="")
if db_score < 1.0:
    print("✓ EXCELLENT (clusters well separated)")
elif db_score < 2.0:
    print("✓ GOOD (acceptable separation)")
else:
    print("⚠ POOR (clusters overlap)")

# Calinski-Harabasz Index (higher is better)
ch_score = calinski_harabasz_score(df_normalized, clusters)
print(f"\nCalinski-Harabasz Index: {ch_score:.3f}")
print(f"  → Interpretation: Higher = Better defined clusters")

# ============================================================================
# 2. CLUSTER DISTRIBUTION
# ============================================================================

print("\n[2] CLUSTER DISTRIBUTION")
print("-" * 70)
cluster_counts = pd.Series(clusters).value_counts().sort_index()
print(cluster_counts)
print(f"\nTotal samples: {len(clusters)}")
print(f"Samples per cluster: {cluster_counts.tolist()}")

# Check if balanced
min_size = cluster_counts.min()
max_size = cluster_counts.max()
balance_ratio = min_size / max_size if max_size > 0 else 0
print(f"Balance ratio: {balance_ratio:.2%}")
if balance_ratio > 0.5:
    print("  → ✓ Clusters are fairly balanced")
else:
    print("  → ⚠ Clusters are imbalanced (some clusters much smaller)")

# ============================================================================
# 3. CLUSTER CHARACTERISTICS (What do they represent?)
# ============================================================================

print("\n[3] WHAT EACH CLUSTER REPRESENTS")
print("-" * 70)

df_analysis = df_clean.copy()
df_analysis['cluster'] = clusters

for cluster_id in sorted(df_analysis['cluster'].unique()):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
    n_samples = len(cluster_data)
    pct = 100 * n_samples / len(df_analysis)
    
    print(f"\nCLUSTER {cluster_id}: {n_samples} samples ({pct:.1f}%)")
    print("-" * 35)
    
    # Key metrics
    avg_speed = cluster_data.get('airspeed', pd.Series([0])).mean()
    avg_alt = cluster_data.get('altitude', pd.Series([0])).mean()
    avg_alt_rate = cluster_data.get('verticalSpeed', pd.Series([0])).mean()
    avg_rpm = cluster_data.get('engineRPM', pd.Series([0])).mean()
    
    print(f"  Airspeed (avg):       {avg_speed:.1f} knots")
    print(f"  Altitude (avg):       {avg_alt:.1f} feet")
    print(f"  Vertical Speed (avg): {avg_alt_rate:.1f} ft/min")
    print(f"  Engine RPM (avg):     {avg_rpm:.0f} rpm")
    
    # Flight phase identification
    print(f"  → Flight Phase: ", end="")
    if avg_speed < 10 and abs(avg_alt_rate) < 50:
        print("🚁 HOVER/STATIONARY")
    elif avg_alt_rate > 100:
        print("📈 CLIMB")
    elif avg_alt_rate < -100:
        print("📉 DESCENT")
    elif avg_speed > 50:
        print("✈️  CRUISE/FAST FLIGHT")
    else:
        print("🔄 TRANSITION/MIXED")

# ============================================================================
# 4. PCA VISUALIZATION
# ============================================================================

print("\n[4] DIMENSIONALITY REDUCTION (PCA)")
print("-" * 70)

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_normalized)

var_exp = pca.explained_variance_ratio_
cum_var = np.cumsum(var_exp)

print(f"PC1 Variance Explained: {var_exp[0]:.1%}")
print(f"PC2 Variance Explained: {var_exp[1]:.1%}")
print(f"Total (2D):             {cum_var[1]:.1%}")

if cum_var[1] > 0.7:
    print("  → ✓ Good 2D representation (>70% variance captured)")
elif cum_var[1] > 0.5:
    print("  → ✓ Acceptable 2D representation (>50% variance)")
else:
    print("  → ⚠ Poor 2D representation (consider 3D visualization)")

# Plot clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', 
                     s=100, alpha=0.6, edgecolors='black', linewidth=1)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='*', s=500, edgecolors='black', linewidth=2, 
           label='Cluster Centers', zorder=5)
plt.xlabel(f'PC1 ({var_exp[0]:.1%} variance)')
plt.ylabel(f'PC2 ({var_exp[1]:.1%} variance)')
plt.title('Flight Clusters Visualization')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# 5. ANOMALY DETECTION
# ============================================================================

print("\n[5] ANOMALY DETECTION (DBSCAN)")
print("-" * 70)

dbscan = DBSCAN(eps=0.5, min_samples=10)
anomalies = dbscan.fit_predict(df_normalized)

n_anomalies = list(anomalies).count(-1)
pct_anomalies = 100 * n_anomalies / len(anomalies)

print(f"Anomalies detected: {n_anomalies} ({pct_anomalies:.1f}%)")
if pct_anomalies < 5:
    print("  → ✓ Normal (few anomalies)")
elif pct_anomalies < 15:
    print("  → ⚠ Moderate anomalies detected")
else:
    print("  → ✗ High anomalies (data quality issue?)")

# ============================================================================
# 6. SUMMARY RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

recommendations = []

if sil_score < 0.3:
    recommendations.append(f"• Silhouette score is low ({sil_score:.3f}). Try different k values.")
if balance_ratio < 0.3:
    recommendations.append("• Clusters are very imbalanced. Consider adjusting parameters.")
if cum_var[1] < 0.6:
    recommendations.append("• PCA 2D doesn't capture enough variance. Use 3D or more features.")
if pct_anomalies > 15:
    recommendations.append("• Too many anomalies. Check data quality or preprocessing.")

if not recommendations:
    print("\n✓ Model looks GOOD! No major issues detected.")
else:
    print("\nPotential improvements:")
    for rec in recommendations:
        print(rec)

print("\n" + "="*70)
print("✓ Evaluation complete!")
print("="*70)
