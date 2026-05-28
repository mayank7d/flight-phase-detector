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
from sklearn.covariance import EllipticCovariance
from sklearn.ensemble import IsolationForest
import seaborn as sns
from scipy.signal import medfilt

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

def preprocess_fdr_data(df, parameters_of_interest=None):
    """
    Preprocess FDR data for ML
    
    Parameters:
    -----------
    df : DataFrame
        Raw FDR data
    parameters_of_interest : list
        List of parameter names to analyze
    """
    
    # Default important helicopter parameters
    if parameters_of_interest is None:
        parameters_of_interest = [
            'airspeed', 'altitude', 'vertical_speed',
            'pitch', 'roll', 'yaw',
            'engine_rpm', 'fuel_flow',
            'oil_temp', 'oil_pressure',
            'main_rotor_rpm', 'tail_rotor_rpm',
            'vibration_x', 'vibration_y', 'vibration_z',
            'g_force_x', 'g_force_y', 'g_force_z'
        ]
    
    # Select available parameters
    available_params = [p for p in parameters_of_interest if p in df.columns]
    df_subset = df[available_params].copy()
    
    print(f"Using {len(available_params)} parameters: {available_params}")
    
    # Handle missing values
    # Option 1: Forward fill for time series data
    df_subset = df_subset.fillna(method='ffill').fillna(method='bfill')
    
    # Option 2: Remove rows with any NaN (if sparse)
    # df_subset = df_subset.dropna()
    
    # Remove outliers using IQR method
    for col in df_subset.columns:
        Q1 = df_subset[col].quantile(0.25)
        Q3 = df_subset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3 IQR for extreme outliers
        upper_bound = Q3 + 3 * IQR
        
        # Mark outliers
        outliers = (df_subset[col] < lower_bound) | (df_subset[col] > upper_bound)
        df_subset.loc[outliers, col] = np.nan
        df_subset[col] = df_subset[col].fillna(method='ffill').fillna(method='bfill')
    
    return df_subset, available_params


def normalize_data(df):
    """
    Normalize data for ML (important for distance-based algorithms)
    
    Use RobustScaler for data with outliers, StandardScaler for normal data
    """
    scaler = RobustScaler()  # Better for outlier-prone FDR data
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
    Create rolling statistics as features
    Useful for capturing flight dynamics
    """
    df_engineered = df.copy()
    
    # Rolling mean (smoothing effect)
    for col in df.columns:
        df_engineered[f'{col}_rolling_mean_{window_size}'] = \
            df[col].rolling(window=window_size, center=True).mean()
        
        # Rolling standard deviation (variability)
        df_engineered[f'{col}_rolling_std_{window_size}'] = \
            df[col].rolling(window=window_size, center=True).std()
    
    # Rate of change
    for col in df.columns:
        df_engineered[f'{col}_delta'] = df[col].diff()
        df_engineered[f'{col}_delta_2nd'] = df[col].diff().diff()
    
    # Fill NaN values created by rolling operations
    df_engineered = df_engineered.fillna(method='bfill')
    
    return df_engineered


# ============================================================================
# 4. CLUSTERING - IDENTIFY FLIGHT PHASES
# ============================================================================

def clustering_analysis(df, n_clusters=4, method='kmeans'):
    """
    Cluster FDR data to identify different flight phases/patterns
    
    Parameters:
    -----------
    df : DataFrame (normalized)
    n_clusters : int
        Number of clusters to identify
    method : str
        'kmeans' or 'dbscan'
    
    Flight phases you might identify:
    - Hover (low speed, stable altitude)
    - Climb (positive vertical speed, increasing altitude)
    - Descent (negative vertical speed, decreasing altitude)
    - Cruise (steady speed, stable altitude)
    - Maneuver (rapid changes in pitch/roll)
    """
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = clusterer.fit_predict(df)
        
        print(f"\nKMeans Clustering Results:")
        print(f"Cluster distribution:\n{pd.Series(clusters).value_counts().sort_index()}")
        
        return clusters, clusterer
    
    elif method == 'dbscan':
        # DBSCAN for anomaly detection + clustering
        clusterer = DBSCAN(eps=0.5, min_samples=10)
        clusters = clusterer.fit_predict(df)
        
        n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_anomalies = list(clusters).count(-1)
        
        print(f"\nDBSCAN Results:")
        print(f"Clusters found: {n_clusters_found}")
        print(f"Anomalies detected: {n_anomalies}")
        print(f"Cluster distribution:\n{pd.Series(clusters).value_counts().sort_index()}")
        
        return clusters, clusterer
    
    return None, None


# ============================================================================
# 5. ANOMALY DETECTION
# ============================================================================

def detect_anomalies(df, method='isolation_forest'):
    """
    Detect unusual flight patterns that might indicate issues
    
    Useful for:
    - Identifying maintenance-related anomalies
    - Finding pilot training events
    - Detecting equipment failures
    """
    
    if method == 'isolation_forest':
        detector = IsolationForest(contamination=0.05, random_state=42)
        anomalies = detector.fit_predict(df)
        anomaly_scores = detector.score_samples(df)
        
        n_anomalies = (anomalies == -1).sum()
        print(f"\nIsolation Forest - Anomalies found: {n_anomalies}")
        
        return anomalies, anomaly_scores, detector
    
    elif method == 'elliptic_covariance':
        # Robust covariance estimation
        detector = EllipticCovariance(random_state=42)
        detector.fit(df)
        anomalies = detector.predict(df)
        anomaly_scores = detector.decision_function(df)
        
        n_anomalies = (anomalies == -1).sum()
        print(f"\nElliptic Covariance - Anomalies found: {n_anomalies}")
        
        return anomalies, anomaly_scores, detector
    
    return None, None, None


# ============================================================================
# 6. DIMENSIONALITY REDUCTION & VISUALIZATION
# ============================================================================

def apply_pca(df, n_components=2):
    """
    Reduce dimensions for visualization
    Good for understanding data structure
    """
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    
    print(f"\nPCA Results:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance explained: {np.cumsum(pca.explained_variance_ratio_)}")
    
    return df_pca, pca


def visualize_clusters(df_pca, clusters, title="Flight Clusters"):
    """Visualize clusters in 2D PCA space"""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/clusters_visualization.png', dpi=300)
    print("Cluster visualization saved!")
    plt.show()


def visualize_anomalies(df_pca, anomalies, title="Anomaly Detection"):
    """Visualize detected anomalies"""
    plt.figure(figsize=(10, 7))
    normal = anomalies == 1
    plt.scatter(df_pca[normal, 0], df_pca[normal, 1], c='blue', alpha=0.5, label='Normal')
    plt.scatter(df_pca[~normal, 0], df_pca[~normal, 1], c='red', marker='X', 
                s=200, label='Anomaly')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/anomalies_visualization.png', dpi=300)
    print("Anomaly visualization saved!")
    plt.show()


# ============================================================================
# 7. ANALYZE CLUSTERS - UNDERSTAND WHAT THEY REPRESENT
# ============================================================================

def analyze_cluster_characteristics(df_original, clusters):
    """
    Understand what each cluster represents
    """
    df_with_clusters = df_original.copy()
    df_with_clusters['cluster'] = clusters
    
    print("\n" + "="*60)
    print("CLUSTER CHARACTERISTICS")
    print("="*60)
    
    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        print(f"\n--- CLUSTER {cluster_id} (n={len(cluster_data)}) ---")
        print(cluster_data.drop('cluster', axis=1).describe().loc[['mean', 'std']])


# ============================================================================
# 8. TIME SERIES ANOMALY DETECTION
# ============================================================================

def detect_time_series_anomalies(df, window_size=50):
    """
    Detect anomalies in time series using deviation from rolling mean
    Useful for sudden changes in parameters
    """
    anomaly_scores = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        rolling_mean = df[col].rolling(window=window_size, center=True).mean()
        rolling_std = df[col].rolling(window=window_size, center=True).std()
        
        # Z-score based on rolling statistics
        z_score = np.abs((df[col] - rolling_mean) / rolling_std)
        anomaly_scores[f'{col}_zscore'] = z_score
    
    # Overall anomaly score (mean across all parameters)
    anomaly_scores['overall_score'] = anomaly_scores.mean(axis=1)
    
    return anomaly_scores


# ============================================================================
# 9. COMPLETE ANALYSIS PIPELINE
# ============================================================================

def run_complete_analysis(file_path, n_clusters=4):
    """
    Run complete unsupervised learning analysis on FDR data
    """
    
    print("="*70)
    print("HELICOPTER FDR - UNSUPERVISED MACHINE LEARNING ANALYSIS")
    print("="*70)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    df_raw = load_fdr_data(file_path)
    
    # 2. Preprocess
    print("\n[2/7] Preprocessing...")
    df_clean, params = preprocess_fdr_data(df_raw)
    
    # 3. Normalize
    print("\n[3/7] Normalizing...")
    df_norm, scaler = normalize_data(df_clean)
    
    # 4. Feature engineering
    print("\n[4/7] Engineering features...")
    df_features = engineer_features(df_clean)
    df_features_norm, _ = normalize_data(df_features)
    
    # 5. PCA for visualization
    print("\n[5/7] Applying PCA...")
    df_pca, pca_model = apply_pca(df_norm, n_components=2)
    
    # 6. Clustering
    print("\n[6/7] Clustering (KMeans)...")
    clusters, kmeans_model = clustering_analysis(df_norm, n_clusters=n_clusters)
    analyze_cluster_characteristics(df_clean, clusters)
    visualize_clusters(df_pca, clusters)
    
    # 7. Anomaly detection
    print("\n[7/7] Detecting anomalies...")
    anomalies, anomaly_scores, isolation_forest = \
        detect_anomalies(df_norm, method='isolation_forest')
    visualize_anomalies(df_pca, anomalies)
    
    # Time series anomalies
    ts_anomalies = detect_time_series_anomalies(df_clean)
    
    # Return all results
    results = {
        'df_raw': df_raw,
        'df_clean': df_clean,
        'df_norm': df_norm,
        'clusters': clusters,
        'anomalies': anomalies,
        'anomaly_scores': anomaly_scores,
        'ts_anomalies': ts_anomalies,
        'pca_data': df_pca,
        'models': {
            'kmeans': kmeans_model,
            'isolation_forest': isolation_forest,
            'pca': pca_model,
            'scaler': scaler
        }
    }
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Replace with your actual FDR data file
    fdr_file = "path/to/your/helicopter_fdr_data.csv"
    
    # Run analysis
    # results = run_complete_analysis(fdr_file, n_clusters=4)
    
    # Access results:
    # clusters = results['clusters']
    # anomalies = results['anomalies']
    # models = results['models']
    
    print("""
    
    ========== QUICK START GUIDE ==========
    
    1. Prepare your CSV file with FDR data
    
    2. Run analysis:
       results = run_complete_analysis('your_file.csv', n_clusters=4)
    
    3. Access results:
       clusters = results['clusters']
       anomalies = results['anomalies']
    
    4. Customize parameters in preprocess_fdr_data():
       - Add/remove helicopter parameters
       - Adjust outlier thresholds
    
    ======================================
    """)
