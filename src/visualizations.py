VISUALIZATION 1 — FLIGHT REGIME VS TIME
# =============================================================================
# VISUALIZATION 1: FLIGHT REGIME VS TIME
# =============================================================================
# PURPOSE:
# Shows how HDBSCAN partitions the flight over time.
#
# HOW TO INTERPRET:
# Long continuous blocks of one cluster usually correspond to a flight phase.
#
# Example:
# Cluster 0 -> Ground
# Cluster 1 -> Climb
# Cluster 2 -> Cruise
# Cluster 3 -> Descent
#
# If clusters rapidly alternate every few samples, clustering is likely unstable.
# =============================================================================

plt.figure(figsize=(15,4))

plt.scatter(
    range(len(df_clean)),
    df_clean['Regime_Cluster'],
    c=df_clean['Regime_Cluster'],
    cmap='viridis',
    s=5
)

plt.xlabel('Time Index')
plt.ylabel('Cluster Label')
plt.title('Flight Regime Evolution Over Time')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


VISUALIZATION 2 — ALTITUDE PROFILE COLORED BY CLUSTER
# =============================================================================
# VISUALIZATION 2: ALTITUDE PROFILE COLORED BY CLUSTER
# =============================================================================
# PURPOSE:
# Most useful plot for identifying flight phases.
#
# HOW TO INTERPRET:
#
# Low altitude cluster
#     -> Ground / Taxi
#
# Rising altitude cluster
#     -> Climb
#
# Constant altitude cluster
#     -> Cruise
#
# Falling altitude cluster
#     -> Descent
# =============================================================================

plt.figure(figsize=(15,5))

plt.scatter(
    range(len(df_clean)),
    df_clean['Zp1'],
    c=df_clean['Regime_Cluster'],
    cmap='viridis',
    s=5
)

plt.xlabel('Time Index')
plt.ylabel('Altitude (Zp1)')
plt.title('Altitude Profile Colored by HDBSCAN Cluster')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


VISUALIZATION 3 — AIRSPEED PROFILE COLORED BY CLUSTER
# =============================================================================
# VISUALIZATION 3: AIRSPEED PROFILE COLORED BY CLUSTER
# =============================================================================
# PURPOSE:
# Helps distinguish hover, climb, cruise and approach.
#
# Cruise:
#     High stable IAS
#
# Hover:
#     Very low IAS
#
# Approach:
#     Gradually decreasing IAS
# =============================================================================

plt.figure(figsize=(15,5))

plt.scatter(
    range(len(df_clean)),
    df_clean['IAS1'],
    c=df_clean['Regime_Cluster'],
    cmap='viridis',
    s=5
)

plt.xlabel('Time Index')
plt.ylabel('IAS')
plt.title('Airspeed Profile Colored by HDBSCAN Cluster')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


VISUALIZATION 4 — ALTITUDE VS VERTICAL SPEED
# =============================================================================
# VISUALIZATION 4: ALTITUDE VS VERTICAL SPEED
# =============================================================================
# PURPOSE:
# Separates climb, cruise and descent very clearly.
#
# Positive VS:
#     Climb
#
# Near-zero VS:
#     Cruise / Hover
#
# Negative VS:
#     Descent
# =============================================================================

plt.figure(figsize=(10,7))

plt.scatter(
    df_clean['Zp1'],
    df_clean['Ver_spd-ADV1'],
    c=df_clean['Regime_Cluster'],
    cmap='viridis',
    s=10,
    alpha=0.7
)

plt.xlabel('Altitude')
plt.ylabel('Vertical Speed')
plt.title('Altitude vs Vertical Speed')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


VISUALIZATION 5 — PITCH VS VERTICAL SPEED
# =============================================================================
# VISUALIZATION 5: PITCH VS VERTICAL SPEED
# =============================================================================
# PURPOSE:
# Useful for distinguishing aggressive climb/descent behaviour.
#
# High pitch + positive VS
#     -> Climb
#
# Negative pitch + negative VS
#     -> Descent
# =============================================================================

plt.figure(figsize=(10,7))

plt.scatter(
    df_clean['Theta1'],
    df_clean['Ver_spd-ADV1'],
    c=df_clean['Regime_Cluster'],
    cmap='viridis',
    s=10,
    alpha=0.7
)

plt.xlabel('Pitch Angle')
plt.ylabel('Vertical Speed')
plt.title('Pitch vs Vertical Speed')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


VISUALIZATION 6 — CLUSTER HEATMAP (MOST IMPORTANT FOR REPORT)
# =============================================================================
# VISUALIZATION 6: CLUSTER FEATURE HEATMAP
# =============================================================================
# PURPOSE:
# Shows average characteristics of each cluster.
#
# This is usually the easiest way to assign names:
#
# Cluster 0:
#     Low Altitude
#     Low IAS
#     -> Ground
#
# Cluster 1:
#     High VS
#     -> Climb
#
# Cluster 2:
#     High Altitude
#     VS ~ 0
#     -> Cruise
# =============================================================================

import seaborn as sns

cluster_means = (
    df_clean[df_clean['Regime_Cluster'] != -1]
    .groupby('Regime_Cluster')[features]
    .mean()
)

plt.figure(figsize=(12,7))

sns.heatmap(
    cluster_means,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Mean Feature Values per HDBSCAN Cluster')

plt.tight_layout()
plt.show()



VISUALIZATION 7 — HDBSCAN CONDENSED TREE
# =============================================================================
# VISUALIZATION 7: HDBSCAN CONDENSED TREE
# =============================================================================
# PURPOSE:
# Shows how HDBSCAN formed clusters.
#
# Large persistent branches:
#     Stable clusters
#
# Tiny short branches:
#     Weak clusters
#
# Useful when tuning:
#     min_cluster_size
#     min_samples
# =============================================================================

clusterer.condensed_tree_.plot()

plt.title('HDBSCAN Condensed Tree')

plt.tight_layout()
plt.show()
