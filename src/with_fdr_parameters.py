"""
Helicopter Flight Data Recorder (FDR) - Unsupervised Machine Learning Analysis
======================================================================
This guide covers clustering, anomaly detection, and pattern recognition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import seaborn as sns
from scipy.signal import medfilt

# ============================================================================
# PARAMETER REGISTRY
# ============================================================================
#
# Each entry maps a canonical name to:
#   - 'columns'     : list of possible column names in your CSV (checked in order)
#   - 'description' : human-readable label
#   - 'required'    : if True, a warning is raised when the parameter is absent
#
# HOW MISSING PARAMETERS ARE HANDLED
# ------------------------------------
# 1. The loader tries every alias in 'columns' against your CSV headers.
# 2. If none match, the parameter is simply skipped — no crash.
# 3. At startup a summary is printed showing which params were FOUND vs MISSING.
# 4. Downstream steps (normalisation, clustering, anomaly detection) work on
#    whatever subset was found, so the pipeline always runs.
# 5. Parameters marked required=True get an extra WARNING in the summary so you
#    know a critical signal is absent.
#
# TO ADD A NEW PARAMETER: append a new dict to PARAMETER_REGISTRY below.
# TO RENAME A CSV COLUMN:  add its exact CSV header to the 'columns' list.

PARAMETER_REGISTRY = [
    # ── Attitude ──────────────────────────────────────────────────────────────
    {
        "name": "theta",
        "description": "Pitch angle (deg)",
        "columns": ["theta", "pitch", "pitch_angle", "PITCH", "THETA"],
        "required": True,
    },
    {
        "name": "phi",
        "description": "Roll angle (deg)",
        "columns": ["phi", "roll", "roll_angle", "ROLL", "PHI"],
        "required": True,
    },
    {
        "name": "psi",
        "description": "Heading / yaw angle (deg)",
        "columns": ["psi", "heading", "yaw", "yaw_angle", "HEADING", "PSI"],
        "required": True,
    },

    # ── Angular rates ─────────────────────────────────────────────────────────
    {
        "name": "p",
        "description": "Roll rate (deg/s)",
        "columns": ["p", "roll_rate", "p_rate", "ROLL_RATE"],
        "required": False,
    },
    {
        "name": "q",
        "description": "Pitch rate (deg/s)",
        "columns": ["q", "pitch_rate", "q_rate", "PITCH_RATE"],
        "required": False,
    },
    {
        "name": "r",
        "description": "Yaw rate (deg/s)",
        "columns": ["r", "yaw_rate", "r_rate", "YAW_RATE"],
        "required": False,
    },

    # ── Velocities ────────────────────────────────────────────────────────────
    {
        "name": "vertical_speed",
        "description": "Vertical speed / rate of climb (ft/min or m/s)",
        "columns": ["vertical_speed", "vspeed", "rate_of_climb", "roc",
                    "VERTICAL_SPEED", "ROC"],
        "required": True,
    },
    {
        "name": "airspeed",
        "description": "Indicated / true airspeed (kts or m/s)",
        "columns": ["airspeed", "ias", "tas", "indicated_airspeed",
                    "true_airspeed", "AIRSPEED", "IAS", "TAS"],
        "required": True,
    },
    {
        "name": "ground_speed",
        "description": "Ground speed (kts or m/s)",
        "columns": ["ground_speed", "groundspeed", "gs", "GS", "GROUND_SPEED"],
        "required": False,
    },

    # ── Altitude / height ─────────────────────────────────────────────────────
    {
        "name": "altitude",
        "description": "Barometric altitude (ft or m)",
        "columns": ["altitude", "baro_altitude", "alt", "ALT", "ALTITUDE"],
        "required": True,
    },
    {
        "name": "radio_altimeter",
        "description": "Radio / radar altimeter (ft or m)",
        "columns": ["radio_altimeter", "radio_alt", "radar_alt", "ra",
                    "RA", "RADIO_ALT", "RADALT"],
        "required": False,
    },

    # ── Rotors & engines ──────────────────────────────────────────────────────
    {
        "name": "nf",
        "description": "Free turbine RPM (% or RPM)",
        "columns": ["nf", "free_turbine_rpm", "nr", "rotor_speed",
                    "main_rotor_rpm", "NF", "NR", "ROTOR_SPEED"],
        "required": True,
    },
    {
        "name": "ng",
        "description": "Gas turbine / gas generator RPM (% or RPM)",
        "columns": ["ng", "gas_turbine_rpm", "n1", "N1", "NG",
                    "GAS_TURBINE_RPM", "engine_rpm"],
        "required": False,
    },

    # ── Accelerations ─────────────────────────────────────────────────────────
    {
        "name": "g_force",
        "description": "Normal load factor / vertical g (g)",
        "columns": ["g_force", "g_accel", "nz", "load_factor",
                    "g_force_z", "G_FORCE", "NZ"],
        "required": False,
    },
    {
        "name": "g_force_x",
        "description": "Longitudinal acceleration (g)",
        "columns": ["g_force_x", "nx", "accel_x", "NX"],
        "required": False,
    },
    {
        "name": "g_force_y",
        "description": "Lateral acceleration (g)",
        "columns": ["g_force_y", "ny", "accel_y", "NY"],
        "required": False,
    },

    # ── Navigation ────────────────────────────────────────────────────────────
    {
        "name": "angle",
        "description": "Generic angle / track angle (deg)",
        "columns": ["angle", "track_angle", "track", "ANGLE", "TRACK"],
        "required": False,
    },
]


# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_fdr_data(file_path):
    """Load FDR data from CSV"""
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

def _resolve_parameters(df_columns, registry=PARAMETER_REGISTRY):
    """
    Match registry entries to actual CSV columns.

    Returns
    -------
    column_map : dict  {canonical_name: actual_csv_column}
    missing    : list  canonical names with no match (warnings printed for required ones)
    """
    df_cols_lower = {c.lower(): c for c in df_columns}   # case-insensitive lookup
    column_map = {}
    missing = []

    print("\n" + "=" * 60)
    print("PARAMETER RESOLUTION SUMMARY")
    print("=" * 60)

    for param in registry:
        matched_col = None
        for alias in param["columns"]:
            # Try exact match first, then case-insensitive
            if alias in df_columns:
                matched_col = alias
                break
            if alias.lower() in df_cols_lower:
                matched_col = df_cols_lower[alias.lower()]
                break

        if matched_col:
            column_map[param["name"]] = matched_col
            print(f"  ✓  {param['name']:20s} → '{matched_col}'  ({param['description']})")
        else:
            missing.append(param["name"])
            tag = "⚠  REQUIRED" if param.get("required") else "–  optional"
            print(f"  {tag:12s} {param['name']:20s}  NOT FOUND  ({param['description']})")

    print(f"\n  Found: {len(column_map)} / {len(registry)} parameters")
    if not column_map:
        raise ValueError("No registered parameters found in the CSV. "
                         "Check column names or update PARAMETER_REGISTRY.")
    return column_map, missing


def preprocess_fdr_data(df, extra_columns=None):
    """
    Preprocess FDR data for ML.

    Parameters
    ----------
    df             : raw DataFrame from load_fdr_data()
    extra_columns  : list of additional CSV column names to include as-is
                     (useful for one-off columns not in the registry)

    Returns
    -------
    df_subset      : cleaned DataFrame using only resolved columns
    column_map     : dict mapping canonical name → actual CSV column used
    missing        : list of canonical names that were absent from the CSV
    """
    # Resolve registry → actual CSV columns
    column_map, missing = _resolve_parameters(df.columns.tolist())

    # Build working DataFrame from resolved columns
    # Use canonical names as column labels for consistency downstream
    df_subset = pd.DataFrame(index=df.index)
    for canonical, csv_col in column_map.items():
        df_subset[canonical] = df[csv_col].copy()

    # Optionally include extra columns the user specifies by CSV name
    if extra_columns:
        for col in extra_columns:
            if col in df.columns:
                df_subset[col] = df[col].copy()
                print(f"  + extra column included: '{col}'")
            else:
                print(f"  ! extra column '{col}' not found in data — skipped")

    # ── Handle missing values (forward-fill, then back-fill for time series) ──
    df_subset = df_subset.ffill().bfill()

    # ── Remove extreme outliers (3 × IQR) and interpolate over them ──────────
    for col in df_subset.columns:
        Q1, Q3 = df_subset[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lo, hi = Q1 - 3 * IQR, Q3 + 3 * IQR
        mask = (df_subset[col] < lo) | (df_subset[col] > hi)
        if mask.any():
            df_subset.loc[mask, col] = np.nan
            df_subset[col] = df_subset[col].ffill().bfill()

    print(f"\nClean data shape: {df_subset.shape}")
    return df_subset, column_map, missing


def normalize_data(df):
    """
    Normalize data for ML.
    RobustScaler is used because FDR data often contains residual outliers.
    """
    scaler = RobustScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    return df_normalized, scaler


# ============================================================================
# 3. FEATURE ENGINEERING FOR TIME SERIES DATA
# ============================================================================

def engineer_features(df, window_size=10):
    """
    Create rolling statistics and rate-of-change features.
    Captures flight dynamics that a single snapshot misses.
    """
    df_engineered = df.copy()

    for col in df.columns:
        # Rolling mean (trend / smoothed signal)
        df_engineered[f'{col}_rmean'] = \
            df[col].rolling(window=window_size, center=True).mean()

        # Rolling std (variability — high during manoeuvres)
        df_engineered[f'{col}_rstd'] = \
            df[col].rolling(window=window_size, center=True).std()

        # First and second derivative (rate of change)
        df_engineered[f'{col}_d1'] = df[col].diff()
        df_engineered[f'{col}_d2'] = df[col].diff().diff()

    df_engineered = df_engineered.bfill()
    return df_engineered


# ============================================================================
# 4. CLUSTERING - IDENTIFY FLIGHT PHASES
# ============================================================================

def clustering_analysis(df, n_clusters=4, method='kmeans'):
    """
    Cluster FDR data to identify different flight phases / patterns.

    Typical clusters for a helicopter:
      0 – Hover          (low airspeed, stable altitude, ~100 % NF)
      1 – Climb          (positive vertical speed, increasing altitude)
      2 – Cruise         (steady airspeed, stable altitude)
      3 – Descent        (negative vertical speed, decreasing altitude)
      (extra) Manoeuvre  (rapid changes in theta / phi / p / q)
    """
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = clusterer.fit_predict(df)
        print(f"\nKMeans – cluster distribution:\n"
              f"{pd.Series(clusters).value_counts().sort_index()}")
        return clusters, clusterer

    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=10)
        clusters = clusterer.fit_predict(df)
        n_found = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_anom  = (clusters == -1).sum()
        print(f"\nDBSCAN – clusters: {n_found}, noise/anomalies: {n_anom}")
        print(pd.Series(clusters).value_counts().sort_index())
        return clusters, clusterer

    return None, None


# ============================================================================
# 5. ANOMALY DETECTION
# ============================================================================

def detect_anomalies(df, method='isolation_forest'):
    """
    Detect unusual flight patterns that may indicate:
      - Mechanical issues (NF/NG exceedances, vibration spikes)
      - Unusual manoeuvres or pilot training events
      - Sensor faults (stuck or frozen values)
    """
    if method == 'isolation_forest':
        detector = IsolationForest(contamination=0.05, random_state=42)
        anomalies = detector.fit_predict(df)
        scores    = detector.score_samples(df)
        print(f"\nIsolation Forest – anomalies: {(anomalies == -1).sum()}")
        return anomalies, scores, detector

    elif method == 'elliptic_covariance':
        detector = EllipticEnvelope(random_state=42)
        detector.fit(df)
        anomalies = detector.predict(df)
        scores    = detector.decision_function(df)
        print(f"\nElliptic Covariance – anomalies: {(anomalies == -1).sum()}")
        return anomalies, scores, detector

    return None, None, None


# ============================================================================
# 6. DIMENSIONALITY REDUCTION & VISUALIZATION
# ============================================================================

def apply_pca(df, n_components=2):
    """Reduce to 2D for visualisation."""
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")
    return df_pca, pca


def visualize_clusters(df_pca, clusters, title="Flight Clusters"):
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters,
                     cmap='viridis', alpha=0.6)
    plt.colorbar(sc, label='Cluster')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title(title)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/clusters_visualization.png', dpi=300)
    print("Cluster visualization saved!")
    plt.show()


def visualize_anomalies(df_pca, anomalies, title="Anomaly Detection"):
    plt.figure(figsize=(10, 7))
    normal = anomalies == 1
    plt.scatter(df_pca[normal,  0], df_pca[normal,  1], c='blue', alpha=0.5, label='Normal')
    plt.scatter(df_pca[~normal, 0], df_pca[~normal, 1], c='red',  marker='X',
                s=200, label='Anomaly')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.legend(); plt.title(title); plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/anomalies_visualization.png', dpi=300)
    print("Anomaly visualization saved!")
    plt.show()


# ============================================================================
# 7. CLUSTER CHARACTERISATION
# ============================================================================

def analyze_cluster_characteristics(df_original, clusters):
    """Print mean & std for each cluster so you can label flight phases."""
    df_c = df_original.copy()
    df_c['cluster'] = clusters

    print("\n" + "=" * 60)
    print("CLUSTER CHARACTERISTICS")
    print("=" * 60)

    for cid in sorted(df_c['cluster'].unique()):
        subset = df_c[df_c['cluster'] == cid]
        print(f"\n--- CLUSTER {cid}  (n={len(subset)}) ---")
        print(subset.drop('cluster', axis=1).describe().loc[['mean', 'std']])


# ============================================================================
# 8. TIME-SERIES ANOMALY DETECTION
# ============================================================================

def detect_time_series_anomalies(df, window_size=50):
    """
    Z-score based anomaly detection using rolling statistics.
    Catches sudden spikes / drops in any channel (e.g. NF drop, pitch exceedance).
    """
    scores = pd.DataFrame(index=df.index)
    for col in df.columns:
        rm  = df[col].rolling(window=window_size, center=True).mean()
        rs  = df[col].rolling(window=window_size, center=True).std()
        scores[f'{col}_zscore'] = np.abs((df[col] - rm) / rs)
    scores['overall_score'] = scores.mean(axis=1)
    return scores


# ============================================================================
# 9. COMPLETE ANALYSIS PIPELINE
# ============================================================================

def run_complete_analysis(file_path, n_clusters=4, extra_columns=None):
    """
    Run the full unsupervised ML pipeline on FDR data.

    Parameters
    ----------
    file_path      : path to the CSV file
    n_clusters     : number of KMeans clusters
    extra_columns  : list of additional CSV column names to include
                     alongside the registered parameters
    """
    print("=" * 70)
    print("HELICOPTER FDR – UNSUPERVISED MACHINE LEARNING ANALYSIS")
    print("=" * 70)

    # 1. Load
    print("\n[1/7] Loading data...")
    df_raw = load_fdr_data(file_path)

    # 2. Preprocess  (auto-resolves parameters; handles missing ones gracefully)
    print("\n[2/7] Preprocessing...")
    df_clean, column_map, missing_params = preprocess_fdr_data(
        df_raw, extra_columns=extra_columns
    )

    if missing_params:
        print(f"\n  ℹ  Continuing without: {missing_params}")

    # 3. Normalise
    print("\n[3/7] Normalising...")
    df_norm, scaler = normalize_data(df_clean)

    # 4. Feature engineering
    print("\n[4/7] Engineering features...")
    df_feat = engineer_features(df_clean)
    df_feat_norm, _ = normalize_data(df_feat)

    # 5. PCA
    print("\n[5/7] Applying PCA...")
    df_pca, pca_model = apply_pca(df_norm, n_components=2)

    # 6. Clustering
    print("\n[6/7] Clustering (KMeans)...")
    clusters, kmeans_model = clustering_analysis(df_norm, n_clusters=n_clusters)
    analyze_cluster_characteristics(df_clean, clusters)
    visualize_clusters(df_pca, clusters)

    # 7. Anomaly detection
    print("\n[7/7] Detecting anomalies...")
    anomalies, anomaly_scores, iso_forest = detect_anomalies(
        df_norm, method='isolation_forest'
    )
    visualize_anomalies(df_pca, anomalies)
    ts_anomalies = detect_time_series_anomalies(df_clean)

    return {
        'df_raw':          df_raw,
        'df_clean':        df_clean,
        'df_norm':         df_norm,
        'column_map':      column_map,
        'missing_params':  missing_params,
        'clusters':        clusters,
        'anomalies':       anomalies,
        'anomaly_scores':  anomaly_scores,
        'ts_anomalies':    ts_anomalies,
        'pca_data':        df_pca,
        'models': {
            'kmeans':         kmeans_model,
            'isolation_forest': iso_forest,
            'pca':            pca_model,
            'scaler':         scaler,
        },
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    fdr_file = "path/to/your/helicopter_fdr_data.csv"

    # results = run_complete_analysis(fdr_file, n_clusters=4)

    # If your CSV has extra columns not in the registry, pass them explicitly:
    # results = run_complete_analysis(fdr_file, extra_columns=["torque", "oat"])

    print("""
    ========== QUICK START GUIDE ==========

    1. Put your CSV path in fdr_file above.

    2. Check column names — the registry tries many aliases automatically.
       If a column is still not found, add its exact CSV header to the
       relevant 'columns' list in PARAMETER_REGISTRY near the top of this file.

    3. Run:
         results = run_complete_analysis('your_file.csv', n_clusters=4)

    4. The console will print which parameters were FOUND vs MISSING so you
       can see exactly what the analysis is working with.

    5. Access results:
         clusters  = results['clusters']
         anomalies = results['anomalies']
         missing   = results['missing_params']

    ======================================
    """)
