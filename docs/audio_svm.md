Conversation with Gemini
Return this text as an md file

(MP2HY0HG)

Davies_boulidin score

Fourier transform breaks down a complex signal into its individual pure frequencies



Imagine blending a banana, strawberries, and milk into a smoothie. The raw audio wave is the smoothie. The Fourier Transform acts like a magical machine that analyzes the drink and tells you exactly how many grams of banana and how many grams of strawberry are inside. [1]

The STFT was invented specifically to fix that timing problem. It brings the element of time back into the equation.Instead of running the Fourier Transform on the entire song at once, the STFT chops the audio into a series of tiny, short, overlapping time blocks (snapshots) and runs the Fourier Transform on each block individually.



MEL SPECTROGRAM

The standard spectrogram shows frequencies on a linear scale (0 Hz, 1000 Hz, 2000 Hz, etc.). However, human ears do not hear the world linearly.Our ears are highly sensitive to small pitch changes at low frequencies, but very bad at detecting pitch changes at high frequencies:You can easily tell the difference between a 500 Hz tone and a 600 Hz tone.You will struggle to tell the difference between a 10,500 Hz tone and a 10,600 Hz tone, even though the physical gap (100 Hz) is exactly the same.The Mel Scale is a mathematical formula that warps frequency numbers so that equal distances on the scale sound like equal distances to a human listener.



A Mel Spectrogram is a standard spectrogram where the vertical Y-axis is squished and stretched to match the Mel Scale.The Low Frequencies (where humans hear fine details) are stretched out to take up more visual space.The High Frequencies (where human hearing is dull) are compressed into less visual space.





But we want machinery sound not human speech recognition

Compare spectrogram vs mel spectrogram vs mfcc spectrogram

Audacity first

It is visual debugger

To understand - Can I see flight phases change

Open audacity -> track -> spectrogram



Look for new frequency bands, energy shifts , sudden transitions

Band energy ratio



Audacity

Ctrl 1 zoom in

Ctrl 3 zoom out

Left of the track switch track -> spectrogram

Track name -> spectrogram settings

FFT SIZE - 4096 ( for helicopter)

What does FFT mean

Small FFT size gives good time resolution and poor frequency resolution

High FFT size gives good freq resolution poor time resolution

Frequency range

Low - > 0

High > 5000

If spectrogram looks crowded shift to 0-2000

If spectrogram looks dark increase gain bright decrease gain

Click selection tool or f1 now click anywhere audacity shows start time end time



Cqt spectrogram

( Constant -Q transform spectrogram) Try to represent audio in a way closer to human hearing

Log scaling



{

"Time": 125,

"altitude": 3500,

"pitch_angle": 4.2,

"roll_angle": 0.8,

"yaw_angle": 12.4,

"regime": "Climb"

}JSON

↓

Read regime string



"Hover"

"Climb"

"Cruise"

"Descent"

"Turn"



↓



LabelEncoder



Hover → 0

Climb → 1

Cruise → 2

...



↓



WAV

↓



MFCC / Spectrogram



↓



SVM / CNN



↓



Predict regime label



Remove fdr features from training

X_fdr = df[fdr_feature_cols].values.astype(np.float32)



X_combined = np.concatenate(

[X_mfcc, X_fdr],

axis=1

)



Replace step 4 with

print("\n" + "═"*60)

print(" STEP 4 / 6 — Using regime labels from JSON")

print("═"*60)



# Audio-only features

X_combined = X_mfcc



print(" Audio feature shape : {}".format(X_combined.shape))

print(" Training target : {}".format(COL_REGIME))



all_feature_names = (

["mfcc_{}".format(i) for i in range(N_MFCC)]

+ (["dmfcc_{}".format(i) for i in range(N_MFCC)]

if INCLUDE_DELTA else [])

+ (["d2mfcc_{}".format(i) for i in range(N_MFCC)]

if INCLUDE_DELTA2 else [])

)

Label remains exactly as they are keep

regimes_raw = df[COL_REGIME].values



le = LabelEncoder()

y = le.fit_transform(regimes_raw)





One more correction is change in train test split

train_test_split(

X_combined,

y,

test_size=TEST_SIZE,

stratify=y,

random_state=RANDOM_STATE,

shuffle=True

)



To

split_idx = int(len(X_combined) * 0.8)



X_train = X_combined[:split_idx]

X_test = X_combined[split_idx:]



y_train = y[:split_idx]

y_test = y[split_idx:]

Since flight data is sequential shuffle can lead to leakage of temporal data

# Helicopter Audio-Based Flight Regime Classification



## Project Goal



The goal of this project is to determine whether helicopter flight regimes can be identified using cockpit audio alone.



Instead of manually labeling audio clips, regime labels are obtained from synchronized Flight Data Recorder (FDR) data.



Examples of flight regimes:



- Hover

- Climb

- Cruise

- Descent

- Turn

- Approach

- Landing



The classifier learns the relationship:



Audio → Flight Regime



---



# High-Level Pipeline



```text

FDR JSON

↓

Regime Labels

(Hover, Climb, Cruise...)



Cockpit WAV Audio

↓

1-second Audio Segments

↓

MFCC Feature Extraction

↓

Feature Scaling

↓

Support Vector Machine (SVM)

↓

Predicted Flight Regime

```



The JSON is used only for:



1. Time alignment

2. Ground truth regime labels



The FDR parameters themselves are NOT used as classifier inputs.



---



# Why This Change Was Made



## Previous Version



The previous code used:



```text

MFCC Features

+

FDR Parameters

+

Regime Labels

```



Input:



```text

[Audio + FDR]

```



Output:



```text

Regime

```



This creates a multimodal classifier.



---



## New Version



The updated code uses:



```text

Audio

↓

MFCC

↓

SVM

↓

Regime

```



while:



```text

FDR

↓

Regime Labels

```



only provides training labels.



This answers a more interesting research question:



Can helicopter sound alone reveal flight regime?



---



# Step 1 — Load JSON



Function:



```python

json.load(...)

pd.DataFrame(...)

```



Purpose:



Load FDR-derived labels.



Example JSON:



```json

{

"Time": 15,

"Altitude": 3200,

"Pitch": 2.3,

"Roll": 0.4,

"regime": "Climb"

}

```



Only two columns are required:



```python

COL_TIME

COL_REGIME

```



Example:



```python

COL_TIME = "Time"

COL_REGIME = "regime"

```



Output:



```text

Time

Regime

```



used later for audio alignment.



---



# Step 2 — Load WAV File



Function:



```python

wavfile.read()

```



Purpose:



Read cockpit audio recording.



Returns:



```python

sample_rate

audio

```



Example:



```python

44100 Hz

```



means:



```text

44100 samples per second

```



---



## Stereo to Mono Conversion



If audio contains:



```text

Left Channel

Right Channel

```



it is converted into:



```text

Mono

```



using:



```python

audio.mean(axis=1)

```



This simplifies processing and reduces feature dimensionality.



---



## Audio Normalization



Purpose:



Convert raw integer amplitudes into:



```text

[-1, +1]

```



range.



Example:



```python

audio /= 32767

```



Benefits:



- Numerical stability

- Consistent feature extraction



---



# Step 3 — Audio Segmentation



Audio is divided into:



```text

1-second windows

```



Example:



```text

0-1 sec

1-2 sec

2-3 sec

...

```



Each window corresponds to:



```text

one JSON row

```



through time alignment.



---



# Why 1-Second Windows?



Because FDR labels are available per second.



Example:



```text

Time 10 → Hover

Time 11 → Hover

Time 12 → Climb

```



Therefore:



```text

Audio 10-11 sec → Hover

Audio 11-12 sec → Hover

Audio 12-13 sec → Climb

```



---



# Step 4 — MFCC Feature Extraction



Most important signal-processing stage.



---



## What is MFCC?



MFCC =



```text

Mel Frequency Cepstral Coefficients

```



MFCC describes:



```text

Shape of sound spectrum

```



instead of raw waveform values.



---



## MFCC Pipeline



```text

Audio

↓

FFT

↓

Power Spectrum

↓

Mel Filter Bank

↓

Log Energy

↓

DCT

↓

MFCC

```



---



## FFT



FFT:



```text

Fast Fourier Transform

```



Converts:



```text

Amplitude vs Time

```



into:



```text

Energy vs Frequency

```



Example:



```text

Rotor frequency

Engine frequency

Gearbox frequency

```



become visible.



---



## Mel Filterbank



Human hearing is logarithmic.



Mel scaling compresses:



```text

High Frequencies

```



and preserves detail in:



```text

Low Frequencies

```



This makes features more robust.



---



## Log Compression



Converts:



```text

Huge Energy Differences

```



into manageable ranges.



Without log:



```text

10000

100

10

```



With log:



```text

4

2

1

```



---



## DCT



DCT:



```text

Discrete Cosine Transform

```



Decorrelates features.



Produces:



```text

MFCC_0

MFCC_1

...

MFCC_12

```



which are easier for machine learning models.



---



# Delta Features



Optional:



```python

INCLUDE_DELTA = True

```



Computes:



```text

ΔMFCC

```



which measures:



```text

Rate of Change

```



---



Example



Instead of:



```text

Current sound

```



model sees:



```text

How sound is changing

```



---



# Delta-Delta Features



Optional:



```python

INCLUDE_DELTA2 = True

```



Computes:



```text

Δ²MFCC

```



which measures:



```text

Acceleration of change

```



---



# Final Audio Feature Vector



If:



```python

N_MFCC = 13

```



then:



```text

13 MFCC

13 Delta

13 Delta-Delta

```



Total:



```text

39 Features / Second

```



---



# Step 5 — Label Encoding



Machine learning cannot use:



```text

Hover

Climb

Cruise

```



directly.



Convert:



```python

LabelEncoder()

```



Example:



```text

Hover → 0

Climb → 1

Cruise → 2

Descent → 3

```



These become target labels.



---



# Step 6 — Feature Scaling



Function:



```python

StandardScaler()

```



Formula:



z = (x − mean)/std



Purpose:



Normalize all features to:



```text

Mean = 0

Std = 1

```



This prevents one feature from dominating others.



Required for SVM.



---



# Step 7 — Train/Test Split



Current approach:



```python

split_idx = int(0.8 * len(X))

```



Training:



```text

First 80%

```



Testing:



```text

Last 20%

```



This preserves time ordering.



Important for flight data.



---



## Why Not Shuffle?



Bad:



```text

Train:

1 sec

100 sec

500 sec



Test:

2 sec

101 sec

501 sec

```



This leaks information.



Time-based split is more realistic.



---



# Step 8 — Support Vector Machine



Classifier:



```python

SVC()

```



---



## What SVM Does



SVM finds:



```text

Decision Boundaries

```



between regimes.



Example:



```text

Hover

|

| Boundary

|

Cruise

```



---



## RBF Kernel



Used:



```python

kernel="rbf"

```



RBF:



```text

Radial Basis Function

```



Allows:



```text

Non-linear boundaries

```



which are common in audio data.



---



## Parameter C



```python

C = 10

```



Controls:



```text

Overfitting vs Underfitting

```



Large C:



```text

Fit training data tightly

```



Small C:



```text

Smoother boundary

```



---



## Gamma



```python

gamma="scale"

```



Controls:



```text

Influence radius

```



of each training sample.



Small gamma:



```text

Smooth regions

```



Large gamma:



```text

Complex boundaries

```



---



## Class Weight



```python

class_weight="balanced"

```



Useful because:



```text

Cruise

```



may occur far more often than:



```text

Landing

```



Prevents bias toward majority classes.



---



# Predictions



The model predicts:



```text

MFCC Features

↓

SVM

↓

Regime

```



Example:



```text

Input:

Audio segment



Output:

Hover

```



---



# Evaluation Metrics



## Accuracy



Formula:



```text

Correct Predictions

-------------------

Total Predictions

```



---



## F1 Score



Balances:



```text

Precision

Recall

```



Useful when class sizes differ.



---



# Gene

rated Plots



## Confusion Matrix



Shows:



```text

True Regime

vs

Predicted Regime

```



Ideal result:



Strong diagonal.



---



## MFCC Heatmap



Shows:



```text

Time

vs

MFCC Coefficients

```



Useful for inspecting extracted features.



---



## PCA Visualization



Projects:



```text

39-dimensional MFCC space

```



into:



```text

2 dimensions

```



for visualization.



Purpose:



See whether regimes form distinct clusters.



---



## Prediction Timeline



Displays:



```text

Time

vs

Predicted Regime

```



with:



```text

Background = True Regime

```



and



```text

Red X = Incorrect Prediction

```



Useful for identifying transitions and failure cases.



---



# Research Question Being Answered



The final system answers:



```text

Can cockpit audio alone identify

helicopter flight regime?

```



More formally:



```text

Audio Signal

↓

MFCC Representation

↓

SVM Classifier

↓

Flight Regime Estimation

```



while using FDR-derived regime labels only as ground truth.

Plaintext
   Cockpit WAV Audio
            ↓
  1-Second Segmentation
            ↓
  MFCC Feature Extraction
            ↓
   Feature Scaling
            ↓
     Classifier (SVM)
            ↓
 Predicted Regime Label
Purpose of FDR Data
The JSON telemetry file is used exclusively for:

Time-alignment with the audio file.

Generating ground-truth training target labels.

Crucial Paradigm Shift: The structural FDR engineering parameters themselves are explicitly omitted from the training inputs. The model must learn to predict flight status relying solely on the acoustic properties of the audio.

4. Pipeline Code Implementations & Core Refinements
Step 1: Target Label Extraction & Encoding
The system ingests the streaming FDR JSON metadata, extracts the flight state, and transforms the textual categorical labels into structured integers.

JSON
{
  "Time": 125,
  "altitude": 3500,
  "pitch_angle": 4.2,
  "roll_angle": 0.8,
  "yaw_angle": 12.4,
  "regime": "Climb"
}
Python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configuration Constants
COL_TIME = "Time"
COL_REGIME = "regime"

# Load labels from telemetry data
regimes_raw = df[COL_REGIME].values

# Encode string labels to categorical integers
le = LabelEncoder()
y = le.fit_transform(regimes_raw)
# Example Mapping: Hover -> 0, Climb -> 1, Cruise -> 2, etc.
Step 2: Audio Loading and Preprocessing
The cockpit audio file is ingested, downmixed to mono, and amplitude-normalized.

Python
from scipy.io import wavfile

# Load raw WAV audio file
sample_rate, audio = wavfile.read("cockpit_audio.wav")

# Stereo to Mono conversion to reduce processing dimensionality
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Audio normalization to bring values into a standard [-1.0, +1.0] range
audio = audio.astype(np.float32) / 32767.0
Step 3: Sequential Audio Segmentation
The continuous mono signal is split into discrete 1-second windows to match the 1 Hz frequency resolution of the FDR telemetry data.

Python
# Example of time-alignment logic mapping to 1Hz data
# Time 10 -> Hover   --> Audio [10-11s] -> Label 0
# Time 11 -> Hover   --> Audio [11-12s] -> Label 0
# Time 12 -> Climb   --> Audio [12-13s] -> Label 1
Step 4: MFCC Feature Isolation (FDR Exclusion Refinement)
The previous pipeline implementation inadvertently combined acoustic arrays with direct FDR parameters, causing data shortcutting and rendering the acoustic objective obsolete. The logic has been completely decoupled to enforce Audio-Only Feature Extraction.

Python
print("\n" + "═"*60)
print("  STEP 4 / 6 — Using regime labels from JSON")
print("═"*60)

# CRITICAL FIX: FDR features (X_fdr) are completely removed from training array inputs
# Previous Code: X_combined = np.concatenate([X_mfcc, X_fdr], axis=1)

# Updated Code: Audio-only features mapping
X_combined = X_mfcc

print("  Audio feature shape : {}".format(X_combined.shape))
print("  Training target     : {}".format(COL_REGIME))

# Dynamically compile feature columns based on feature expansion flags
all_feature_names = (
    ["mfcc_{}".format(i) for i in range(N_MFCC)]
    + (["dmfcc_{}".format(i) for i in range(N_MFCC)] if INCLUDE_DELTA else [])
    + (["d2mfcc_{}".format(i) for i in range(N_MFCC)] if INCLUDE_DELTA2 else [])
)
Step 7: Sequential Train/Test Split (Anti-Leakage Refinement)
Standard random splitting strategies shuffle continuous temporal observations across groups. Because flight sequences contain heavy temporal dependencies, uniform random shuffling leaks adjacent frame information, creating artificially inflated metrics.

The pipeline has been updated to use a structural Sequential Time-Series Split.

Python
# PREVIOUS BUGGY CODE (Removed to prevent temporal data leakage):
# train_test_split(X_combined, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE, shuffle=True)

# UPDATED TIME-SERIES COMPLIANT EXECUTION:
split_idx = int(len(X_combined) * 0.8)

# Continuous block allocation preserves genuine chronological sequence boundaries
X_train = X_combined[:split_idx]
X_test  = X_combined[split_idx:]

y_train = y[:split_idx]
y_test  = y[split_idx:]
5. MFCC Feature Extraction Pipeline
The Mel Frequency Cepstral Coefficients (MFCC) condense high-dimensional spectral shapes into compact acoustic descriptors.

Feature Generation Pipeline
Plaintext
Audio Waveform → FFT → Power Spectrum → Mel Filter Bank → Log Energy → DCT → MFCCs
FFT (Fast Fourier Transform): Transforms the 1-second time-domain segment into an explicit frequency spectrum, exposing foundational engine components, rotor blade passing frequencies, and gearbox acoustic peaks.

Mel Filterbank: Warps the linear frequency metrics using a collection of triangular bandpass filters to match human auditory resolution limits.

Log Compression: Converts wide-ranging physical power scaling onto a logarithmic decibel scale, matching biological volume perception and reducing absolute variance.

DCT (Discrete Cosine Transform): Decorrelates the overlapping filterbank energies, generating compact coefficients (MFCC 
0
​
 ,MFCC 
1
​
 ,…,MFCC 
12
​
 ) optimized for linear and kernel classifiers.

Delta Framework Expansion
Delta Features (Velocity): Evaluates the first-order derivative (

ΔMFCC
) representing the instantaneous rate of spectral shift over time.

Delta-Delta Features (Acceleration): Evaluates the second-order derivative (

Δ 
2
 MFCC
) measuring acoustic transition acceleration.

Vector Profile: Assuming 13 primary MFCCs with both Delta and Delta-Delta extensions active, each 1-second segment yields a complete 39-dimensional feature vector.

6. Model Training & Evaluation Metrics
Feature Scaling
Because Support Vector Machine (SVM) objective functions rely heavily on distance-based calculations, a StandardScaler must be applied to transform the features to have a mean of 0 and a standard deviation of 1 (z= 
σ
x−μ
​
 ). This keeps high-energy coefficients from dominating lower-amplitude variables during optimization.

Support Vector Machine (SVM) Classification Settings
Kernel Function: kernel="rbf" (Radial Basis Function) is used to map non-linear spectral properties into a higher-dimensional space where classes become linearly separable.

Regularization Parameter (C=10): Balances classification accuracy on the training data against boundary smoothness to prevent overfitting.

Gamma Parameter (gamma="scale"): Dynamically scales the structural influence radius of individual support vectors based on feature variance.

Imbalance Compensation (class_weight="balanced"): Automatically adjusts weights inversely proportional to class frequencies. This prevents majority flight regimes (like Cruise) from biasing the model against brief, critical operational states (like Landing or Approach).

Diagnostic Artifacts & Visualizations
Confusion Matrix: Evaluates target classification precision across all classes. A strong diagonal trend indicates clean class separation.

MFCC Heatmap: Plots Coefficient Index vs. Time to visually audit the quality of the extracted audio features.

PCA Cluster Analysis: Reduces the 39-dimensional feature space down to 2 dimensions to visually confirm whether different flight regimes form distinct acoustic clusters.

Prediction Timeline: Plots time against predicted regimes, overlaying the ground-truth flight states and flagging incorrect classification events with a Red X to help isolate failure modes during flight state transitions.
"""

with open("helicopter_classification.md", "w", encoding="utf-8") as f:
f.write(markdown_content)

print("Markdown file successfully generated."