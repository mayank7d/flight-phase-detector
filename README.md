# flight-phase-detector
# Helicopter FDR - Unsupervised ML Quick Reference Guide

## TL;DR - Start Here

**Your Task:** Use unsupervised ML on helicopter FDR data to find patterns without labels

**Best Approach:** 
1. Use **K-Means clustering** (4 clusters) to identify flight phases
2. Use **Isolation Forest** for anomaly detection
3. Use **PCA** for visualization

**Expected Output:** Clusters representing hover/climb/cruise/descent phases + anomalies detected

---

## Top Parameters to Include (In Priority Order)

| Priority | Parameter | Why |
|----------|-----------|-----|
| 🔴 Critical | Airspeed | Main differentiator between flight phases |
| 🔴 Critical | Altitude | Fundamental flight parameter |
| 🔴 Critical | Vertical Speed | Shows climb vs descent vs level flight |
| 🟠 High | Pitch, Roll, Yaw | Aircraft attitude (important for anomalies) |
| 🟠 High | Engine RPM | Operational state |
| 🟠 High | Fuel Flow | Power indicator |
| 🟡 Medium | Vibration (X,Y,Z) | Structural health anomalies |
| 🟡 Medium | G-Forces (X,Y,Z) | Load patterns |
| 🟡 Medium | Oil Temp/Pressure | Engine health |
| 🟢 Optional | Rotor RPM, Pedal angles | Add if available |

---

## Quick Decision Tree: Which Algorithm to Use?

```
START HERE
    ↓
Do you want to identify FLIGHT PHASES?
├─ YES → Use K-MEANS CLUSTERING (start with k=4)
│        "Normal operations that fall into distinct groups"
│
└─ NO, find UNUSUAL FLIGHTS?
   ├─ YES → Use ISOLATION FOREST
   │        "Finds outliers (anomalies) automatically"
   │
   └─ Also find SUDDEN CHANGES during flight?
      └─ YES → Use TIME SERIES ANOMALY DETECTION
               "Detect rapid parameter changes"
```

---

## 5-Minute Minimal Model

Copy-paste this to get started immediately:

```python
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load
df = pd.read_csv('your_data.csv')

# 2. Clean
df = df.fillna(method='ffill').dropna()

# 3. Select key columns
cols = ['airspeed', 'altitude', 'vertical_speed', 'pitch', 'roll', 'engine_rpm']
df = df[cols]

# 4. Normalize
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)

# 5. Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# 6. Visualize
pca = PCA(n_components=2)
data_2d = pca.fit_transform(df_scaled)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters)
plt.show()

# Done! Each cluster represents a flight state
```

---

## Data Preprocessing Checklist

- [ ] Load CSV file
- [ ] Check for missing values (handle with forward/backward fill)
- [ ] Remove extreme outliers (values > Q3 + 3*IQR)
- [ ] Select important parameters (use table above)
- [ ] Normalize data (RobustScaler is best for FDR data)
- [ ] Verify data range looks reasonable

```python
# Quick check:
print(df.describe())  # Should see reasonable ranges
print(df.isnull().sum())  # Should be near zero
```

---

## Algorithm Quick Comparison

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **K-Means** | Flight phases | Fast, interpretable, simple | Need to specify k in advance |
| **DBSCAN** | Anomalies | Finds outliers automatically | Parameter tuning tricky |
| **Isolation Forest** | Anomalies | Works well with outliers | Less interpretable |
| **Hierarchical** | Relationships | Shows structure | Slow on large data |
| **Autoencoder** | Complex patterns | Most powerful | Requires neural networks, more data |

---

## Parameter Tuning Quick Guide

### K-Means: Choosing Number of Clusters

```python
from sklearn.metrics import silhouette_score

# Try different k values
best_k = 4
best_score = 0

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42)
    score = silhouette_score(df_scaled, km.fit_predict(df_scaled))
    print(f"k={k}: score={score:.3f}")
    if score > best_score:
        best_score = score
        best_k = k

# Use best_k
```

**Expected silhouette scores:**
- \> 0.5: Excellent clustering
- 0.3-0.5: Good clustering
- < 0.3: Poor clustering

### DBSCAN: Tuning eps and min_samples

```
Start: eps=0.5, min_samples=10

If too many anomalies detected:
  → Decrease eps (0.3) 
  → Decrease min_samples (5)

If too few anomalies detected:
  → Increase eps (0.7)
  → Increase min_samples (20)
```

---

## Expected Results & Interpretation

### What K-Means Should Find

If you use 4 clusters with helicopter data, you'll likely see:

| Cluster | Characteristics | Flight Phase |
|---------|-----------------|--------------|
| 0 | Low speed, low vertical rate, stable | **HOVER** |
| 1 | Positive vertical rate, climbing | **CLIMB** |
| 2 | High speed, stable altitude | **CRUISE** |
| 3 | Negative vertical rate | **DESCENT** |

(Exact mapping depends on your data and parameters)

### Verification

```python
# Check what each cluster represents:
for cluster_id in range(4):
    cluster_data = df[clusters == cluster_id]
    print(f"Cluster {cluster_id}:")
    print(cluster_data[['airspeed', 'altitude', 'vertical_speed']].mean())
```

Should see patterns that make physical sense!

---

## Evaluation Metrics

### Use These to Assess Quality

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette Score (higher is better, -1 to 1)
sil_score = silhouette_score(df_scaled, clusters)
print(f"Silhouette: {sil_score:.3f}")  # Should be > 0.3

# Davies-Bouldin Index (lower is better)
db_score = davies_bouldin_score(df_scaled, clusters)
print(f"Davies-Bouldin: {db_score:.3f}")  # Should be < 2

# Cluster distribution
print("Samples per cluster:", np.bincount(clusters))
# Should be relatively balanced
```

---

## Visualization Must-Haves

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. PCA 2D scatter of clusters
pca = PCA(n_components=2)
data_2d = pca.fit_transform(df_scaled)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Clusters in PCA Space')
plt.show()

# 2. Anomaly visualization
anomalies = isolation_forest.fit_predict(df_scaled)
plt.scatter(data_2d[anomalies==1, 0], data_2d[anomalies==1, 1], 
           c='blue', label='Normal')
plt.scatter(data_2d[anomalies==-1, 0], data_2d[anomalies==-1, 1], 
           c='red', marker='X', s=200, label='Anomaly')
plt.legend()
plt.show()

# 3. Parameter correlation with cluster
for param in ['airspeed', 'altitude', 'vertical_speed']:
    print(f"{param} by cluster:")
    print(df.groupby('cluster')[param].mean())
```

---

## Common Mistakes to Avoid

❌ **Don't:** Use raw unscaled data
✓ **Do:** Normalize with RobustScaler

❌ **Don't:** Include too many irrelevant parameters
✓ **Do:** Focus on 10-15 most important parameters

❌ **Don't:** Forget to visualize results
✓ **Do:** Always make 2D/3D plots

❌ **Don't:** Use random_state=None (non-reproducible)
✓ **Do:** Always set random_state=42

❌ **Don't:** Trust results without domain validation
✓ **Do:** Verify clusters make physical sense

---

## Feature Engineering Tricks (If Simple Model Doesn't Work)

```python
# Add temporal features
df['airspeed_change'] = df['airspeed'].diff()
df['altitude_change'] = df['altitude'].diff()
df['speed_rolling_avg'] = df['airspeed'].rolling(window=10).mean()

# Add derived features
df['power_efficiency'] = df['airspeed'] / (df['fuel_flow'] + 1e-6)
df['maneuver_intensity'] = (abs(df['pitch_change']) + abs(df['roll_change'])) / 2

# Add interaction terms
df['speed_x_altitude'] = df['airspeed'] * df['altitude']

# Then normalize and re-cluster
```

---

## Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Clusters too scattered | Poor normalization | Use RobustScaler, not StandardScaler |
| All points in one cluster | Parameters not varied enough | Add more distinctive parameters |
| Silhouette score < 0.3 | Bad k value | Try different k (3, 5, 6) |
| No anomalies found | Contamination too low | Isolation Forest: use contamination=0.10 |
| Results not reproducible | Random seed not fixed | Always use random_state=42 |
| Visualization shows no pattern | PCA variance too low | Include more differentiating parameters |

---

## Output Files to Create

For your internship deliverable:

1. **cluster_assignments.csv** - Original data + cluster IDs
2. **anomaly_scores.csv** - Data points + anomaly scores
3. **cluster_analysis.txt** - Mean characteristics per cluster
4. **visualizations.png** - PCA plots + anomaly plots
5. **model_summary.pdf** - Algorithm choice, metrics, interpretation

---

## Timeline for Your Internship

**Week 1:** Explore data, implement basic K-Means
**Week 2:** Add anomaly detection, feature engineering
**Week 3:** Fine-tune parameters, create visualizations
**Week 4:** Documentation, interpretation, final report

---

## Key Takeaways

1. **Start with K-Means** - Most intuitive for flight data
2. **Use 4 clusters** - Hover, Climb, Cruise, Descent
3. **Always normalize** - Use RobustScaler
4. **Always visualize** - PCA 2D plots most important
5. **Validate with domain knowledge** - Do clusters make physical sense?
6. **Document everything** - Your code, parameters, findings

---

## Resources

- Scikit-learn clustering docs: https://scikit-learn.org/stable/modules/clustering.html
- PCA tutorial: https://scikit-learn.org/stable/modules/decomposition.html
- Anomaly detection: https://scikit-learn.org/stable/modules/ensemble.html#isolation-forest

---

## Questions to Ask HAL Engineers

1. What is the sampling rate of your FDR data?
2. What are the physical limits for each parameter?
3. Are there known issues or maintenance events in your dataset?
4. What would be most valuable: finding flight patterns or detecting anomalies?
5. Do you have any labeled examples of anomalous flights?

---

**Good luck! You've got this! 🚁**
