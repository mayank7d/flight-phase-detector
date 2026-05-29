"""
Helicopter Flight Data Recorder (FDR) - Unsupervised Machine Learning Analysis
======================================================================
This guide covers clustering, anomaly detection, and pattern recognition

Data input: JSON-LD (.jsonld)
----------------------------------------------------------------------
NOTE: A separate conversion script is needed to produce the JSON-LD file
      from your raw Excel FDR export before running this pipeline.
      See the stub at the bottom of this file:
          excel_to_jsonld()  ← YOU NEED TO WRITE / FILL IN THIS FUNCTION
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

# ============================================================================
# PARAMETER REGISTRY
# ============================================================================
#
# Each entry maps a canonical name to:
#   - 'columns'     : possible key names inside each JSON-LD observation object
#                     (checked in order, case-insensitive)
#   - 'description' : human-readable label
#   - 'required'    : if True, a warning is raised when the parameter is absent
#
# HOW MISSING PARAMETERS ARE HANDLED
# ------------------------------------
# 1. The loader tries every alias against the keys present in the JSON-LD data.
# 2. If none match, the parameter is simply skipped — no crash.
# 3. At startup a summary prints which params were FOUND vs MISSING.
# 4. Downstream steps work on whatever subset was found.
# 5. Parameters marked required=True get an extra WARNING in the summary.
#
# TO ADD A NEW PARAMETER: append a new dict to PARAMETER_REGISTRY below.
# TO RENAME A JSON-LD KEY:  add its exact key name to the 'columns' list.

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
# EXCEL → JSON-LD CONVERSION  (STUB — YOU NEED TO WRITE THIS)
# ============================================================================
#
# TODO: Write a conversion script that reads your Excel FDR export and
#       produces a JSON-LD file in the format expected by load_fdr_jsonld()
#       below before running this ML pipeline.
#
# Expected JSON-LD structure this pipeline reads:
#
# {
#   "@context": {
#     "fdr":  "https://example.org/fdr#",
#     "xsd":  "http://www.w3.org/2001/XMLSchema#",
#     "theta": "fdr:theta",
#     "phi":   "fdr:phi",
#     ... (one entry per parameter)
#   },
#   "@graph": [
#     {
#       "@type": "fdr:Observation",
#       "timestamp": "2024-01-01T00:00:00Z",
#       "theta": 2.1,
#       "phi":  -0.5,
#       "psi":  185.3,
#       "nf":   102.4,
#       "airspeed": 80.0,
#       ... (any subset of parameters is fine — missing ones are handled)
#     },
#     { ... },   ← one object per time-step / sample
#     ...
#   ]
# }
#
# SUGGESTED LIBRARIES FOR THE CONVERSION SCRIPT:
#   pip install openpyxl pandas pyld
#
# ROUGH OUTLINE OF excel_to_jsonld():
#   1. pd.read_excel(excel_path)           — load the sheet
#   2. rename columns to match aliases     — align to PARAMETER_REGISTRY keys
#   3. df.to_dict(orient='records')        — list of row dicts
#   4. wrap in @context + @graph skeleton  — add JSON-LD envelope
#   5. json.dump(...)                      — write to .jsonld file

def excel_to_jsonld(excel_path, jsonld_path):
    """
    STUB — convert an Excel FDR export to a JSON-LD file.

    YOU NEED TO IMPLEMENT THIS based on your specific Excel layout.
    See the format notes in the block comment above.
    """
    raise NotImplementedError(
        "excel_to_jsonld() is not yet implemented.\n"
        "Write this function to convert your Excel FDR export to JSON-LD,\n"
        f"then call: excel_to_jsonld('{excel_path}', '{jsonld_path}')\n"
        "before running run_complete_analysis()."
    )


# ============================================================================
# 1. DATA LOADING FROM JSON-LD
# ============================================================================

def load_fdr_jsonld(file_path):
    """
    Load FDR data from a JSON-LD file and return a flat DataFrame.

    The function handles two common JSON-LD layouts:

    Layout A — @graph array (preferred):
        { "@context": {...}, "@graph": [ {obs1}, {obs2}, ... ] }

    Layout B — top-level array:
        [ {obs1}, {obs2}, ... ]

    Each observation object becomes one row in the DataFrame.
    '@type', '@id', and '@context' keys are automatically dropped.
    'timestamp' (if present) is parsed as a datetime index.
    """
    print(f"Loading JSON-LD file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # ── Extract the list of observation records ───────────────────────────────
    if isinstance(raw, dict):
        if "@graph" in raw:
            # Layout A: standard JSON-LD with @graph
            records = raw["@graph"]
        else:
            # Single observation wrapped in a dict — treat as one-row dataset
            records = [raw]
    elif isinstance(raw, list):
        # Layout B: bare array of observations
        records = raw
    else:
        raise ValueError(
            "Unrecognised JSON-LD structure. Expected a dict with '@graph' "
            "or a top-level array of observation objects."
        )

    print(f"  Found {len(records)} observation records in JSON-LD file")

    # ── Flatten records to a DataFrame ───────────────────────────────────────
    # Strip JSON-LD meta-keys that are not data fields
    _skip = {"@type", "@id", "@context"}
    clean_records = [
        {k: v for k, v in rec.items() if k not in _skip}
        for rec in records
    ]

    df = pd.DataFrame(clean_records)

    # ── Parse timestamp as index if present ──────────────────────────────────
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
            print("  Timestamp column parsed and set as index")
        except Exception as e:
            print(f"  Warning: could not parse timestamp column — {e}")

    # ── Convert all data columns to numeric (coerce non-numeric to NaN) ───────
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Data shape after loading: {df.shape}")
    print(f"  Columns found: {df.columns.tolist()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    return df


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

def _resolve_parameters(df_columns, registry=PARAMETER_REGISTRY):
    """
    Match registry entries to actual DataFrame columns (from JSON-LD keys).

    Returns
    -------
    column_map : dict  {canonical_name: actual_column}
    missing    : list  canonical names with no match
    """
    df_cols_lower = {c.lower(): c for c in df_columns}
    column_map = {}
    missing = []

    print("\n" + "=" * 60)
    print("PARAMETER RESOLUTION SUMMARY")
    print("=" * 60)

    for param in registry:
        matched_col = None
        for alias in param["columns"]:
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
        raise ValueError(
            "No registered parameters found in the JSON-LD data. "
            "Check the key names in your JSON-LD file and update "
            "PARAMETER_REGISTRY if needed."
        )
    return column_map, missing


def preprocess_fdr_data(df, extra_columns=None):
    """
    Preprocess FDR data for ML.

    Parameters
    ----------
    df             : raw DataFrame from load_fdr_jsonld()
    extra_columns  : list of additional column names to include as-is

    Returns
    -------
    df_subset  : cleaned DataFrame with canonical column names
    column_map : dict mapping canonical name → actual column used
    missing    : list of canonical names absent from the data
    """
    column_map, missing = _resolve_parameters(df.columns.tolist())

    # Build working DataFrame using canonical names
    df_subset = pd.DataFrame(index=df.index)
    for canonical, src_col in column_map.items():
        df_subset[canonical] = df[src_col].copy()

    # Include any extra columns the caller requested
    if extra_columns:
        for col in extra_columns:
            if col in df.columns:
                df_subset[col] = df[col].copy()
                print(f"  + extra column included: '{col}'")
            else:
                print(f"  ! extra column '{col}' not found in data — skipped")

    # Fill gaps (forward-fill then back-fill — good for time-series dropouts)
    df_subset = df_subset.ffill().bfill()

    # Remove extreme outliers (3 × IQR) and interpolate over them
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
    Normalise with RobustScaler (median + IQR — resistant to residual outliers).
    Must be called before any distance-based ML algorithm.
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
        df_engineered[f'{col}_rmean'] = \
            df[col].rolling(window=window_size, center=True).mean()
        df_engineered[f'{col}_rstd'] = \
            df[col].rolling(window=window_size, center=True).std()
        df_engineered[f'{col}_d1'] = df[col].diff()
        df_engineered[f'{col}_d2'] = df[col].diff().diff()

    df_engineered = df_engineered.bfill()
    return df_engineered


# ============================================================================
# 4. CLUSTERING — IDENTIFY FLIGHT PHASES
# ============================================================================

def clustering_analysis(df, n_clusters=4, method='kmeans'):
    """
    Cluster FDR data to identify different flight phases / patterns.

    Typical clusters for a helicopter:
      0 – Hover          (low airspeed, stable altitude, ~100 % NF)
      1 – Climb          (positive vertical speed, increasing altitude)
      2 – Cruise         (steady airspeed, stable altitude)
      3 – Descent        (negative vertical speed, decreasing altitude)
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
      - Mechanical issues (NF/NG exceedances)
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
    """Reduce to 2D for visualisation only (not used for clustering itself)."""
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
    plt.scatter(df_pca[normal,  0], df_pca[normal,  1],
                c='blue', alpha=0.5, label='Normal')
    plt.scatter(df_pca[~normal, 0], df_pca[~normal, 1],
                c='red', marker='X', s=200, label='Anomaly')
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
    Rolling Z-score anomaly detection — catches sudden spikes within a phase
    (e.g. brief NF drop during cruise) that global detectors might miss.
    """
    scores = pd.DataFrame(index=df.index)
    for col in df.columns:
        rm = df[col].rolling(window=window_size, center=True).mean()
        rs = df[col].rolling(window=window_size, center=True).std()
        scores[f'{col}_zscore'] = np.abs((df[col] - rm) / rs)
    scores['overall_score'] = scores.mean(axis=1)
    return scores


# ============================================================================
# 9. COMPLETE ANALYSIS PIPELINE
# ============================================================================

def run_complete_analysis(file_path, n_clusters=4, extra_columns=None):
    """
    Run the full unsupervised ML pipeline on FDR data from a JSON-LD file.

    Parameters
    ----------
    file_path     : path to the .jsonld file
    n_clusters    : number of KMeans clusters (try 4–6 for typical flights)
    extra_columns : list of JSON-LD key names to include alongside
                    the registered parameters (e.g. ["torque", "oat"])
    """
    print("=" * 70)
    print("HELICOPTER FDR – UNSUPERVISED MACHINE LEARNING ANALYSIS")
    print("=" * 70)

    # 1. Load JSON-LD
    print("\n[1/7] Loading JSON-LD data...")
    df_raw = load_fdr_jsonld(file_path)

    # 2. Preprocess (resolves parameter names; handles missing ones gracefully)
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

    # 5. PCA (for visualisation only)
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
            'kmeans':           kmeans_model,
            'isolation_forest': iso_forest,
            'pca':              pca_model,
            'scaler':           scaler,
        },
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    # ── STEP 0 (one-time): convert your Excel export to JSON-LD ──────────────
    #
    # TODO: implement excel_to_jsonld() above, then uncomment:
    # excel_to_jsonld(
    #     excel_path="path/to/your/fdr_export.xlsx",
    #     jsonld_path="path/to/your/fdr_data.jsonld",
    # )

    # ── STEP 1: run the ML pipeline on the JSON-LD file ──────────────────────
    fdr_file = "path/to/your/fdr_data.jsonld"

    # results = run_complete_analysis(fdr_file, n_clusters=4)

    # To include extra parameters not in the registry:
    # results = run_complete_analysis(fdr_file, extra_columns=["torque", "oat"])

    print("""
    ========== QUICK START GUIDE ==========

    1. Implement excel_to_jsonld() to convert your Excel FDR export.
       See the format comment above that function for the expected structure.

    2. Run the conversion (once per flight file):
         excel_to_jsonld('export.xlsx', 'fdr_data.jsonld')

    3. Run the analysis:
         results = run_complete_analysis('fdr_data.jsonld', n_clusters=4)

    4. The console will print which parameters were FOUND vs MISSING.
       If a key is not found, add its exact name to the relevant
       'columns' list in PARAMETER_REGISTRY at the top of this file.

    5. Access results:
         clusters  = results['clusters']
         anomalies = results['anomalies']
         missing   = results['missing_params']

    ======================================
    """)
