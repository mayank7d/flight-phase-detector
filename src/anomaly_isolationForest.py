from sklearn.ensemble import IsolationForest
# =============================================================================
# ANOMALY DETECTION USING ISOLATION FOREST
# =============================================================================
#
# PURPOSE:
# Detect unusual flight records that do not resemble normal behaviour.
#
# Unlike HDBSCAN:
# - Does NOT create clusters
# - Only identifies anomalies
#
# Typical anomalies:
# - Sensor spikes
# - Corrupted data
# - Unusual maneuvers
# - Rare flight conditions
#
# =============================================================================

print("\nRunning Isolation Forest...")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

df_clean['IF_Anomaly'] = iso_forest.fit_predict(scaled_data)n_estimators=200
contamination=0.02
#Try:

#0.01
#0.02
#0.05

#and compare.


n_anomalies = (df_clean['IF_Anomaly'] == -1).sum()

print(f"Anomalies detected: {n_anomalies}")
print(f"Percentage anomalous: {100*n_anomalies/len(df_clean):.2f}%")
# =============================================================================
# ALTITUDE PROFILE WITH ISOLATION FOREST ANOMALIES
# =============================================================================

plt.figure(figsize=(15,5))

normal_mask = df_clean['IF_Anomaly'] == 1
anomaly_mask = df_clean['IF_Anomaly'] == -1

plt.scatter(
    df_clean.index[normal_mask],
    df_clean.loc[normal_mask, 'Zp1'],
    s=5,
    alpha=0.5,
    label='Normal'
)

plt.scatter(
    df_clean.index[anomaly_mask],
    df_clean.loc[anomaly_mask, 'Zp1'],
    s=30,
    color='red',
    label='Anomaly'
)

plt.xlabel("Time")
plt.ylabel("Altitude")
plt.title("Isolation Forest Anomalies")

plt.legend()
plt.grid(alpha=0.3)

plt.show()
# =============================================================================
# COMPARISON: HDBSCAN vs ISOLATION FOREST
# =============================================================================

comparison = pd.crosstab(
    df_clean['Regime_Cluster'] == -1,
    df_clean['IF_Anomaly'] == -1,
    rownames=['HDBSCAN Noise'],
    colnames=['Isolation Forest Anomaly']
)

print(comparison)

"""
How to interpret

Suppose you get:

                 IF Normal   IF Anomaly

HDBSCAN Normal      9500         100

HDBSCAN Noise        200         150

Interpretation:

Both algorithms agree
150 points

Highly suspicious.

Investigate first.

HDBSCAN says noise

but

Isolation Forest says normal
200 points

Likely rare flight phases.

Maybe:

Takeoff
Landing
Hover

that occur infrequently.

Isolation Forest says anomaly

but

HDBSCAN says normal
100 points

Interesting.

Means:

Inside cluster
BUT
statistically unusual

Often sensor spikes."""

#View the actual anomalous rows 
anomalies = df_clean[df_clean['IF_Anomaly'] == -1]

print(
    anomalies[
        [
            'Zp1',
            'IAS1',
            'Ver_spd-ADV1',
            'Theta1',
            'NF'
        ]
    ].head(20)
)
