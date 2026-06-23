# Helicopter Flight Regime Discovery and Audio-Based Regime Classification

## Technical Project Report

### Author

Mayank Dewan

### Organization

Hindustan Aeronautics Limited (HAL)

---

# 1. Project Motivation

Modern helicopters generate large quantities of flight data through onboard Flight Data Recorders (FDRs). These recordings contain parameters such as:

* Altitude
* Pitch Angle
* Roll Angle
* Yaw Angle
* Vertical Speed
* Pitch Rate
* Roll Rate
* Yaw Rate

Traditionally, engineers inspect these signals manually to understand flight behavior.

The primary objective of this project is to automatically identify meaningful flight regimes from FDR data and subsequently determine whether helicopter cockpit audio alone can be used to recognize these regimes.

Examples of flight regimes include:

* Hover
* Climb
* Cruise
* Descent
* Turn
* Approach
* Landing

The project is divided into two major phases:

1. Flight Regime Discovery using FDR data
2. Audio-Based Flight Regime Classification

---

# 2. Phase I: Flight Regime Discovery from FDR Data

## Objective

Automatically identify operational flight regimes without manually specifying them.

Instead of:

"Tell the algorithm what Hover is"

the approach becomes:

"Allow the algorithm to discover recurring flight behaviors."

---

# 3. Data Preparation

The flight data consists of time-series measurements.

Typical features:

```text
Altitude
Pitch Angle
Roll Angle
Yaw Angle
Vertical Speed
Pitch Rate
Roll Rate
Yaw Rate
Altitude Rate of Change
Vertical Speed Rate of Change
```

These parameters were normalized and converted into a structured CSV/JSON format.

---

# 4. LSTM Autoencoder

## Why Use an Autoencoder?

The original FDR dataset contains multiple correlated parameters.

The objective is:

```text
High-Dimensional Flight Data
            ↓
Compressed Representation
            ↓
Meaningful Flight Behavior Encoding
```

---

## Autoencoder Structure

Input:

```text
30-second flight window
```

↓

Encoder

↓

Latent Representation

↓

Decoder

↓

Reconstructed Flight Window

---

## Why LSTM?

Flight data is sequential.

Traditional feedforward networks ignore temporal relationships.

LSTM (Long Short-Term Memory) networks are designed to learn:

* Temporal dependencies
* Sequential behavior
* Dynamic state evolution

The LSTM encoder learns a compressed latent representation describing aircraft behavior during a flight segment.

---

# 5. Latent Space Generation

After training:

```text
Flight Window
      ↓
Encoder
      ↓
Latent Vector
```

Each flight segment becomes a point in latent space.

Instead of clustering raw flight parameters, clustering is performed on latent vectors.

This significantly improves cluster quality.

---

# 6. HDBSCAN Clustering

## Why HDBSCAN?

Unlike K-Means:

* Does not require specifying number of clusters
* Handles irregular cluster shapes
* Identifies noise
* Works well with real flight data

Pipeline:

```text
FDR
 ↓
LSTM Autoencoder
 ↓
Latent Space
 ↓
HDBSCAN
 ↓
Flight Regimes
```

---

# 7. Clustering Results

Example output:

```text
Cluster 0
Cluster 1
Cluster 2
...
Cluster 9

Noise
```

Noise points typically correspond to:

* Flight transitions
* Unusual maneuvers
* Sensor fluctuations
* Regime boundaries

---

# 8. Visualization Improvements

Several visualization issues were identified and corrected.

Original plots:

* Difficult to interpret
* Combined unrelated information
* Poor cluster visibility

Improved design:

1. PCA Cluster Projection
2. Altitude vs Time Colored by Regime
3. Feature Timelines
4. Cluster Distribution Analysis

Each cluster receives a unique color and legend.

---

# 9. Significance of Regime Discovery

Regime discovery enables:

### Flight Segmentation

Automatically divide flights into:

```text
Hover
Climb
Cruise
Descent
```

---

### Anomaly Detection

Identify:

```text
Unknown Cluster
```

or

```text
Outlier Behavior
```

that may indicate abnormal flight conditions.

---

### Pilot Behavior Analysis

Compare operational styles across pilots.

---

### Training Label Generation

Regime labels become ground truth for future supervised learning models.

This directly enables Phase II.

---

# 10. Phase II: Audio-Based Regime Classification

## Objective

Determine whether cockpit audio contains sufficient information to identify flight regime.

Question:

Can helicopter sound alone reveal aircraft state?

---

# 11. Audio Processing Pipeline

Input:

```text
WAV Recording
```

↓

1-second audio windows

↓

MFCC Extraction

↓

Machine Learning Classifier

↓

Flight Regime Prediction

---

# 12. Regime Labels

Initially the idea was:

```text
Audio
+
FDR Features
↓
Classifier
```

However this was modified.

Final design:

```text
FDR
 ↓
Regime Labels

Audio
 ↓
MFCC
 ↓
SVM
 ↓
Regime
```

The JSON now provides:

* Time alignment
* Ground-truth labels

The FDR parameters themselves are NOT used as classifier inputs.

This creates a true audio-only classifier.

---

# 13. MFCC Features

MFCC:

```text
Mel Frequency Cepstral Coefficients
```

Pipeline:

```text
Audio
 ↓
FFT
 ↓
Mel Filter Bank
 ↓
Log Energy
 ↓
DCT
 ↓
MFCC
```

MFCCs summarize spectral shape and are widely used in speech and acoustic classification.

---

# 14. Additional Audio Features

The system also supports:

### Delta MFCC

Measures change in MFCC values.

### Delta-Delta MFCC

Measures rate of change of Delta MFCC.

Final feature vector:

```text
13 MFCC
13 Delta
13 Delta-Delta

Total = 39 Features
```

---

# 15. Why SVM?

Support Vector Machine was chosen because:

* Strong performance on small datasets
* Effective in high-dimensional spaces
* Robust against overfitting

---

# 16. Understanding SVM

SVM attempts to separate classes using an optimal decision boundary.

Goal:

```text
Maximum Margin Separation
```

between classes.

Only points nearest to the boundary matter.

These are called:

```text
Support Vectors
```

---

# 17. Kernel Functions

Real-world data is rarely linearly separable.

Kernel functions transform data into a space where separation becomes easier.

---

# 18. RBF Kernel

RBF:

```text
Radial Basis Function
```

Measures similarity between samples.

Benefits:

* Handles complex class boundaries
* Suitable for acoustic features
* Strong default choice for MFCC classification

---

# 19. Overfitting

Overfitting occurs when a model memorizes training data rather than learning general patterns.

Symptoms:

```text
Training Accuracy = Very High
Test Accuracy = Low
```

Mitigation:

* Separate training and testing flights
* Cross-validation
* Simpler models
* Regularization

---

# 20. Major Evaluation Improvements

Several improvements were made:

### Multi-Flight Split

Instead of:

```text
Same Flight
↓
Train/Test
```

Now:

```text
Flight 20
Flight 21
 ↓
Train

Flight 22
 ↓
Test
```

This provides a more realistic evaluation.

---

### GroupKFold Validation

Validation now occurs across flights rather than random seconds.

---

# 21. Baseline Models

Added for scientific rigor.

### Random Guess

Lower bound performance.

### Majority Class Predictor

Predicts most common regime.

### Random Forest

Tree-based baseline.

### SVM

Primary classifier.

Comparison allows meaningful interpretation of results.

---

# 22. Random Forest Baseline

Random Forest:

```text
Many Decision Trees
 ↓
Voting
 ↓
Prediction
```

Provides a strong non-linear baseline for comparison against SVM.

---

# 23. Audacity and Spectrogram Analysis

Audacity is used for exploratory analysis.

Goals:

* Identify visible regime transitions
* Observe harmonic structures
* Inspect rotor-related frequencies
* Study broadband noise behavior

Spectrogram:

```text
X = Time
Y = Frequency
Color = Energy
```

---

# 24. Frequency Analysis Hypothesis

Potential investigation:

Extract:

```text
f1
f2
f3
```

dominant spectral ridges.

Then correlate with:

* Yaw Rate
* Pitch
* Roll
* Altitude
* Vertical Speed

This may reveal physically meaningful acoustic signatures.

---

# 25. Future Work

## Stage 1

MFCC → SVM → Regime

(Current)

---

## Stage 2

Spectrogram → CNN → Regime

Compare against MFCC/SVM.

---

## Stage 3

Audio Regime Discovery

```text
Audio
 ↓
CNN Encoder
 ↓
Latent Space
 ↓
HDBSCAN
```

Compare audio clusters with FDR clusters.

---

## Stage 4

Multimodal Learning

```text
Audio
+
FDR
 ↓
Fusion Network
 ↓
Regime
```

---

## Stage 5

Acoustic State Estimation

Predict:

* Pitch
* Roll
* Yaw
* Altitude Band

directly from audio.

---

# 26. Final Vision

The long-term objective is to build an intelligent rotorcraft state understanding system capable of:

```text
Flight Data
+
Audio
+
Machine Learning
```

to achieve:

* Automatic flight segmentation
* Regime recognition
* Anomaly detection
* Pilot behavior analysis
* Health monitoring
* Acoustic state estimation

Ultimately the project seeks to answer:

"How much information about helicopter state is encoded in sound?"
