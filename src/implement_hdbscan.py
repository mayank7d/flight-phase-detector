import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── PACKAGES NEEDED (install at home before bringing to office) ───────────────
# pip install hdbscan
# Everything else (pandas, matplotlib, sklearn) is pre-installed in Anaconda.
# No openpyxl needed — we export to CSV which opens directly in Excel.
# ─────────────────────────────────────────────────────────────────────────────

# --- STEP 1: LOAD THE JSON FILE ---
json_filename = 'fdr_data.json'
print(f"Loading data from {json_filename}...")
df_raw = pd.read_json(json_filename)

# --- STEP 2: SELECT BASE FEATURES AND ENGINEER RATE FEATURES ---
# Rate features (diff) capture how fast each parameter is changing.
# This helps HDBSCAN separate phases like cruise (altitude_rate ≈ 0)
# from climb (altitude_rate > 0) even when the raw altitude values overlap.
# We use diff() instead of rolling mean because rolling requires knowing
# your data's sample rate — diff() works regardless.
base_features = ['Zp1', 'IAS1', 'Ver_spd-ADV1', 'Theta1', 'NF']

df_raw['Altitude_rate'] = df_raw['Zp1'].diff()    # how fast altitude is changing
df_raw['IAS_rate']      = df_raw['IAS1'].diff()   # how fast airspeed is changing
df_raw['Pitch_rate']    = df_raw['Theta1'].diff() # how fast pitch is changing

features = base_features + ['Altitude_rate', 'IAS_rate', 'Pitch_rate']

df_clean = df_raw.dropna(subset=features).copy()
print(f"Processing {len(df_clean)} records across {len(features)} features "
      f"({len(base_features)} base + 3 rate features).")

# --- STEP 3: SCALE THE DATA ---
print("Scaling features...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[features])

# --- STEP 4: RUN HDBSCAN CLUSTERING ---
# min_cluster_size : raise if too many tiny clusters, lower if too many noise points
# min_samples      : raise for tighter clusters, lower to be more generous
print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=10)
# min_cluster_size=200 works well for this synthetic dataset (2000 rows, 5 clean phases).
# On your real FDR data, retune this — a good starting rule is ~1-2% of your total row count.
# If you get too many clusters: raise it. Too many noise points (-1): lower it.
df_clean['Regime_Cluster'] = clusterer.fit_predict(scaled_data)

n_clusters = df_clean['Regime_Cluster'].nunique() - (1 if -1 in df_clean['Regime_Cluster'].values else 0)
n_noise    = (df_clean['Regime_Cluster'] == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points (labelled -1): {n_noise} ({100 * n_noise / len(df_clean):.1f}% of data)")

# --- STEP 5: SILHOUETTE SCORE — cluster quality check ---
# Measures how well-separated the clusters are.
# > 0.5  = very good separation
# 0.25–0.5 = reasonable
# < 0.25 = clusters are overlapping / weak
cluster_mask = df_clean['Regime_Cluster'] != -1
noise_mask   = ~cluster_mask

if cluster_mask.sum() > 0:
    score = silhouette_score(
        scaled_data[cluster_mask],
        df_clean.loc[cluster_mask, 'Regime_Cluster']
    )
    print(f"\nSilhouette Score: {score:.3f}  "
          f"({'Good' if score > 0.5 else 'Reasonable' if score > 0.25 else 'Weak — consider tuning min_cluster_size'})")

# --- STEP 6: PCA FOR VISUALIZATION ONLY (8D → 3D) ---
print("\nApplying PCA for 3D visualization...")
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)
print(f"Variance explained — PC1: {pca.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%  "
      f"PC3: {pca.explained_variance_ratio_[2]*100:.1f}%  "
      f"Total: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# --- STEP 7: 3D VISUALIZATION ---
print("Generating 3D visualization...")
fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    pca_data[cluster_mask, 0],
    pca_data[cluster_mask, 1],
    pca_data[cluster_mask, 2],
    c=df_clean.loc[cluster_mask, 'Regime_Cluster'],
    cmap='viridis', s=30, alpha=0.7,
    label='Flight regime'
)
ax.scatter(
    pca_data[noise_mask, 0],
    pca_data[noise_mask, 1],
    pca_data[noise_mask, 2],
    c='lightgrey', s=15, alpha=0.4, marker='x',
    label=f'Noise ({n_noise} pts)'
)
ax.set_title('Helicopter Flight Regimes — HDBSCAN (3D PCA View)')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.legend(loc='upper left')
fig.colorbar(scatter, ax=ax, label='Regime Cluster', shrink=0.6, pad=0.1)
plt.tight_layout()

# --- STEP 8: CLUSTER PROFILING (MIN, MAX, MEAN, STD) ---
# Inspect the mean values per cluster to map them to flight phases.
# Example pattern to look for:
#   Low Zp1 + Low IAS + Ver_spd ≈ 0            → Hover / Ground
#   Increasing Zp1 + Positive Ver_spd + High Theta → Climb
#   High Zp1 + High IAS + Ver_spd ≈ 0           → Cruise
#   Decreasing Zp1 + Negative Ver_spd           → Descent
print("\n" + "=" * 70)
print("       DETAILED CLUSTER STATISTICAL PROFILES (MIN, MAX, MEAN, STD)")
print("=" * 70)

cluster_profiles = (
    df_clean[cluster_mask]
    .groupby('Regime_Cluster')[features]
    .agg(['count', 'min', 'max', 'mean', 'std'])
)

for feature in features:
    print(f"\n>>> Parameter: {feature} <<<")
    if feature == features[0]:
        print(f"    Points per cluster: {cluster_profiles[feature]['count'].to_dict()}")
    print(cluster_profiles[feature].drop(columns='count').round(2))
    print("-" * 50)

print("=" * 70 + "\n")

# --- STEP 9: SAVE TO CSV ---
# Opens directly in Excel — no extra packages needed.
# Original fdr_data.json is never modified.
output_file = 'fdr_data_with_clusters.csv'
df_clean.to_csv(output_file, index=False)
print(f"Saved: {output_file}  →  open in Excel to see the Regime_Cluster column.")

# Show the plot
plt.show()


# =============================================================================
#  # STEP 10: SUPERVISED PHASE LABELING — RANDOM FOREST (OPTIONAL)
# =============================================================================
# This step is independent of everything above.
# Your  Random Forest model predicts a named flight phase
# (e.g. "Hover", "Climb", "Cruise", "Descent", "Approach") for each row,
# using the same base parameters as features.
#
# TO USE THIS STEP:
#   1. Get the trained model file (.pkl) from supervised project.
#   2. Put it in the same folder as this script.
#   3. Update RF_MODEL_FILE below with its exact filename.
#   4. Check that RF_FEATURES matches exactly what their model was trained on
#      (column names must be identical — ask your colleague).
#   5. Uncomment the block and run.
#
# OUTPUT:
#   - A new column 'RF_Phase' added to every row with the predicted phase name.
#   - A cross-tabulation table showing how HDBSCAN clusters map to RF phases —
#     this is the key insight: do unsupervised clusters align with supervised labels?
#   - CSV re-saved with both Regime_Cluster and RF_Phase columns.
# =============================================================================

# ── UNCOMMENT EVERYTHING BELOW WHEN YOU HAVE THE MODEL FILE ──────────────────

# import joblib
#
# RF_MODEL_FILE = 'flight_phase_rf_model.pkl'   # ← update with actual filename
#
# # Features the RF was trained on — must match your colleague's code exactly
# # (column names must be identical, same order)
# RF_FEATURES = ['Zp1', 'IAS1', 'Ver_spd-ADV1', 'Theta1', 'NF']
#
# print("\n" + "=" * 70)
# print("STEP 10: SUPERVISED PHASE LABELING — RANDOM FOREST")
# print("=" * 70)
#
# # Load the trained model
# rf_model = joblib.load(RF_MODEL_FILE)
# print(f"Model loaded: {RF_MODEL_FILE}")
#
# # Predict a named phase for every row
# # We use base features only — RF was trained on raw parameters, not rate features
# df_clean['RF_Phase'] = rf_model.predict(df_clean[RF_FEATURES])
#
# print(f"\nPhase distribution (Random Forest predictions):")
# print(df_clean['RF_Phase'].value_counts())
#
# # --- COMPARISON: HDBSCAN clusters vs RF phases ---
# # Read this table to understand how the two approaches relate:
# #
# #   CLEAN mapping (good):
# #     Cluster 0 → 95% Cruise       means HDBSCAN found the same thing RF did
# #     Cluster 1 → 90% Climb        clean, operationally meaningful cluster
# #
# #   MESSY mapping (needs investigation):
# #     Cluster 2 → 40% Cruise, 35% Descent, 25% Climb
# #     means HDBSCAN is grouping by a mathematical pattern that doesn't
# #     correspond to a single flight phase — consider retuning min_cluster_size
# #
# #   Cluster -1 (noise) being mixed across phases is expected and normal
#
# print("\nHDBSCAN Cluster vs RF Phase — cross-tabulation:")
# print(pd.crosstab(
#     df_clean['Regime_Cluster'],
#     df_clean['RF_Phase'],
#     margins=True,
#     margins_name='TOTAL'
# ))
#
# # Re-save CSV with both columns to see both
# df_clean.to_csv(output_file, index=False)
# print(f"\nCSV re-saved with RF_Phase column added: {output_file}")
# print("=" * 70 + "\n")
