# Helicopter FDR Unsupervised Learning Model - Complete Strategy Guide

## Overview
Your goal is to apply unsupervised ML to helicopter FDR data to extract meaningful patterns without labeled data. This is ideal for discovering flight patterns, anomalies, and operational insights.

---

## PHASE 1: DATA UNDERSTANDING & PARAMETER SELECTION

### Essential Parameters to Include

**Core Flight Dynamics (High Priority)**
- `Airspeed` - Key differentiator between hover/cruise/descent
- `Altitude` - Fundamental flight parameter
- `Vertical_Speed` (Rate of Climb/Descent) - Shows climb vs descent vs level flight
- `Pitch_Angle` - Aircraft attitude
- `Roll_Angle` - Aircraft attitude
- `Yaw_Angle` - Direction control
- `Ground_Speed` - For wind assessment

**Engine & Power System (High Priority)**
- `Engine_RPM` - Operational state indicator
- `Fuel_Flow` - Real-time consumption
- `Oil_Temperature` - Engine health indicator
- `Oil_Pressure` - Engine health indicator
- `Engine_Torque` - Power output

**Rotor System (Important for Helicopters)**
- `Main_Rotor_RPM` - Operational speed
- `Tail_Rotor_RPM` - Control reference
- `Collective_Pitch` - Lift control input
- `Cyclic_Pitch_Lateral` - Lateral control
- `Cyclic_Pitch_Longitudinal` - Longitudinal control

**Structural Health (High Priority)**
- `Vibration_X, Y, Z` - Anomalies appear as unusual vibration
- `G_Force_X, Y, Z` - Acceleration patterns
- `Hydraulic_Pressure` - System operation

**Lower Priority (Add if Available)**
- Outside Air Temperature
- Wind Speed/Direction
- Battery Voltage
- Electrical System Status

### Data Quality Checks
```
Before modeling, verify:
✓ Time synchronization (all parameters time-aligned)
✓ Sampling rate consistent (e.g., 1Hz, 10Hz)
✓ No major gaps or data corruption
✓ Sensor calibration (values within expected ranges)
✓ Timestamp format consistent
```

---

## PHASE 2: PREPROCESSING STRATEGY

### Step 1: Data Cleaning
```python
# Missing values handling
- For < 1% missing: Interpolate
- For 1-5% missing: Forward fill or backward fill
- For > 5% missing: Consider removing that flight/parameter

# Outlier detection
- Use IQR method (Interquartile Range): Remove values > Q3 + 3*IQR
- Use domain knowledge: Airspeed > max for aircraft = error
- Use Mahalanobis distance for multivariate outliers
```

### Step 2: Normalization
```
Use RobustScaler (better for FDR data with possible outliers)
- Less sensitive to extreme values
- Good for distance-based algorithms (KMeans, DBSCAN)

Avoid StandardScaler if you have sudden maneuvers/anomalies
```

### Step 3: Segmentation Options

**Option A: Use entire flight**
- Pros: See overall flight pattern
- Cons: Mixing different phases

**Option B: Segment by time windows (recommended)**
- Segment into 30-60 second windows
- Process each window separately
- Identify patterns at different temporal scales

**Option C: Segment by flight phase**
- Use altitude/speed to identify phases first
- Then cluster within each phase
- Better for phase-specific insights

---

## PHASE 3: CHOOSING UNSUPERVISED LEARNING METHODS

### Method 1: K-Means Clustering ⭐ **RECOMMENDED FOR FIRST MODEL**

**What it does:** Groups similar flight states into clusters

**Best for identifying:**
- Flight phases (hover, climb, cruise, descent)
- Operational patterns
- Different pilot behaviors
- Equipment configurations

**Implementation:**
```python
from sklearn.cluster import KMeans

# Choose n_clusters (3-5 typical for helicopters):
# 3: Hover, Cruise, Maneuver
# 4: Hover, Climb, Cruise, Descent  
# 5: Add "Anomalous" operations

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

# Then analyze: What's the average airspeed in each cluster?
# What's the average altitude? Pattern emerges naturally.
```

**Advantages:**
- Fast and interpretable
- Works well with your data volume
- Easy to explain results
- Natural cluster interpretation

**Parameters to tune:**
- `n_clusters`: Start with 4, try 3-6
- `init`: Use 'k-means++'
- `n_init`: Use 10-20 for stability

---

### Method 2: DBSCAN (Density-Based Clustering)

**What it does:** Finds clusters of arbitrary shape, labels outliers as noise

**Best for identifying:**
- Anomalies mixed with normal operations
- Non-spherical flight patterns
- Automatic outlier detection

**Implementation:**
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
clusters = dbscan.fit_predict(normalized_data)

# Returns:
# Cluster labels: 0, 1, 2, ... (normal clusters)
# Label -1: Anomalies/noise
```

**When to use:**
- If you expect unusual operations mixed in
- Good for finding maintenance-required flights
- Better than KMeans for anomaly detection

**Parameter tuning:**
- `eps`: Distance threshold (0.3-1.0). Smaller = stricter, more outliers
- `min_samples`: Min points in cluster (5-20). Larger = fewer clusters

---

### Method 3: Hierarchical Clustering

**What it does:** Creates tree of nested clusters at different scales

**Best for:**
- Understanding relationships between flight states
- Multi-scale analysis
- Creating dendrograms for visualization

```python
from scipy.cluster.hierarchy import linkage, dendrogram

linkage_matrix = linkage(normalized_data, method='ward')
dendrogram(linkage_matrix)  # Visualize cluster hierarchy
```

---

### Method 4: Isolation Forest - Anomaly Detection

**What it does:** Identifies points that are hard to isolate (anomalies)

**Best for:**
- Finding unusual flights/maneuvers
- Maintenance prediction
- Safety anomalies

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05)
anomalies = iso_forest.fit_predict(normalized_data)
# Returns: 1 (normal), -1 (anomaly)
```

---

### Method 5: Autoencoders (Advanced)

**What it does:** Neural network learns compressed representation of data

**Best for:**
- More complex pattern discovery
- Better anomaly detection
- If you have >100K data points

```python
# Requires TensorFlow/PyTorch - more complex setup
# Recommended after you master simpler methods
```

---

## PHASE 4: FEATURE ENGINEERING FOR TIME SERIES

### Temporal Features
```python
# 1. Rate of change (derivatives)
df['altitude_change'] = df['altitude'].diff()
df['speed_change'] = df['airspeed'].diff()

# 2. Rolling statistics (captures trends)
df['speed_rolling_mean_30s'] = df['airspeed'].rolling(30).mean()
df['altitude_rolling_std_30s'] = df['altitude'].rolling(30).std()

# 3. Acceleration (2nd derivative)
df['altitude_acceleration'] = df['altitude'].diff().diff()

# 4. Ratios and interactions
df['power_efficiency'] = df['airspeed'] / df['fuel_flow']
df['climb_to_power_ratio'] = df['vertical_speed'] / df['engine_rpm']
```

### Flight Phase Indicators
```python
# Create features that help identify phases naturally
df['is_climbing'] = (df['vertical_speed'] > 100).astype(int)
df['is_descending'] = (df['vertical_speed'] < -100).astype(int)
df['is_hovering'] = (df['airspeed'] < 10).astype(int)

# Maneuver intensity (combined attitude changes)
df['maneuver_intensity'] = (abs(df['pitch_change']) + 
                           abs(df['roll_change'])) / 2
```

---

## PHASE 5: MODEL BUILDING WORKFLOW

### Step-by-Step Implementation

```
1. START SIMPLE
   ├─ Use basic parameters (speed, altitude, altitude_rate, pitch, roll)
   ├─ Apply KMeans with n_clusters=4
   └─ Visualize with PCA

2. ADD COMPLEXITY
   ├─ Include engine/rotor parameters
   ├─ Add temporal features
   └─ Try DBSCAN with KMeans

3. ANALYZE RESULTS
   ├─ What does each cluster represent?
   ├─ Are anomalies meaningful?
   └─ Do results align with domain knowledge?

4. REFINE
   ├─ Adjust parameters
   ├─ Include new features
   ├─ Try different methods
   └─ Document findings
```

---

## PHASE 6: EVALUATION & INTERPRETATION

### Silhouette Score
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(normalized_data, clusters)
# Range: -1 to 1
# > 0.5: Good separation
# 0.3-0.5: Acceptable
# < 0.3: Poor clustering
```

### Davies-Bouldin Index
```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(normalized_data, clusters)
# Lower is better (0 = perfect)
```

### Visual Interpretation
```python
# Reduce to 2D for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_2d = pca.fit_transform(normalized_data)

# Plot clusters
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters)
plt.show()

# Check variance explained
print(pca.explained_variance_ratio_)  # Should be >80% combined
```

### Domain Interpretation
```python
# What does each cluster represent?
for cluster_id in unique_clusters:
    cluster_data = data[clusters == cluster_id]
    
    print(f"Cluster {cluster_id}:")
    print(f"  Avg Speed: {cluster_data['airspeed'].mean()}")
    print(f"  Avg Altitude: {cluster_data['altitude'].mean()}")
    print(f"  Avg Vertical Speed: {cluster_data['vertical_speed'].mean()}")
    print(f"  Engine RPM: {cluster_data['engine_rpm'].mean()}")
    
    # This should naturally reveal: hover, climb, cruise, descent
```

---

## PHASE 7: EXPECTED OUTCOMES

### What You Should Discover

**Natural Clustering:**
- **Cluster 1 (Hover):** Low airspeed, low altitude change, high rotor RPM, steady engine
- **Cluster 2 (Climb):** Positive vertical speed, moderate airspeed, high engine power
- **Cluster 3 (Cruise):** High airspeed, stable altitude, moderate engine power
- **Cluster 4 (Descent/Maneuver):** Negative vertical speed or high attitude changes

**Anomalies Identified:**
- Unusual vibration patterns (sensor malfunction or structural issue)
- Engine parameters outside normal ranges
- Unsafe attitude combinations
- Rapid unplanned changes

---

## PHASE 8: TROUBLESHOOTING

### Problem: Clusters don't make sense

**Solution:**
1. Check data preprocessing (missing values, scaling)
2. Verify parameter selection (remove irrelevant ones)
3. Try different n_clusters
4. Use feature importance analysis
5. Plot individual parameters colored by cluster

### Problem: All points labeled as anomalies (DBSCAN)

**Solution:**
1. Decrease eps value (0.3 instead of 0.5)
2. Increase min_samples (20 instead of 10)
3. Normalize data better
4. Check for data distribution issues

### Problem: PCA visualization doesn't show clear clusters

**Solution:**
1. Use TSNE or UMAP instead (better for high-dimensional data)
2. Include more features that differentiate clusters
3. Increase PCA components to 3D
4. Verify clustering quality with silhouette score

---

## PHASE 9: DELIVERABLES FOR YOUR INTERNSHIP

### Create These Outputs:

1. **Data Analysis Report**
   - Summary statistics
   - Data quality assessment
   - Parameter correlations

2. **Clustering Results**
   - Number of clusters found
   - Silhouette/Davies-Bouldin scores
   - Interpretation of each cluster
   - 2D/3D visualization

3. **Anomaly Detection Report**
   - Anomalies found (count + percentage)
   - Anomaly characteristics
   - Potential causes (maintenance? extreme weather?)

4. **Model Parameters Documentation**
   - Which parameters used
   - Preprocessing steps
   - Algorithm choices + justification
   - Hyperparameters used

5. **Code & Reproducibility**
   - Well-commented Python code
   - Can run with new FDR data
   - Documentation of dependencies

6. **Insights & Recommendations**
   - What patterns discovered
   - Which flights are anomalous
   - Recommendations for maintenance
   - Suggestions for further analysis

---

## RECOMMENDED APPROACH FOR YOUR INTERNSHIP

### Week 1-2: Foundation
- Load and explore your FDR dataset
- Implement basic preprocessing
- Run KMeans clustering with 4 clusters
- Visualize results

### Week 3: Enhancement
- Add feature engineering
- Try DBSCAN for anomaly detection
- Evaluate multiple clustering metrics
- Create comprehensive visualizations

### Week 4: Analysis & Refinement
- Interpret cluster meanings
- Investigate anomalies in detail
- Fine-tune hyperparameters
- Prepare final deliverables

---

## Key Python Libraries You'll Need

```
pandas              # Data manipulation
numpy               # Numerical computing
scikit-learn        # ML algorithms
matplotlib          # Plotting
seaborn             # Statistical plots
scipy               # Scientific computing
jupyter             # Interactive notebooks (recommended)

# Optional (for advanced visualization)
plotly              # Interactive plots
umap                # Better dimensionality reduction than PCA
```

---

## Final Tips for Success

1. **Start Simple:** Get KMeans working first, then add complexity
2. **Visualize Everything:** Plots reveal more than numbers
3. **Know Your Data:** Spend time understanding what each parameter means
4. **Validate Results:** Do clusters make physical sense?
5. **Document Well:** Your future self will thank you
6. **Ask Domain Experts:** HAL engineers can validate your findings
7. **Iterate:** First model rarely perfect - improve gradually
8. **Test on Multiple Flights:** Ensure reproducibility across different flights

Good luck with your internship! This is a real-world ML problem with practical applications.
