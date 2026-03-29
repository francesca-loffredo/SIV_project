# SIV Project - Upper Body Posture Analysis

A posture classification system that detects upper body posture from webcam input using MediaPipe pose estimation and machine learning. The system classifies five posture types: lean backward (TLB), lean forward (TLF), lean left (TLL), lean right (TLR), and upright (TUP).

## Pipeline Overview

The notebook is organized into four main sections:

**1. Dataset Analysis** - Exploration of the raw landmark data: class distribution, landmark quality checks, geometric consistency verification.

**2. Feature Engineering** - Extraction of 13 geometric features from body landmarks (head, shoulders, hips). Features are divided into 9 primary features (head and shoulder based) and 4 support features (trunk based). Distances and offsets are normalized by shoulder width. A per-subject z-score normalization with a standard deviation floor is applied before training.

**3. ML Model Training** - Leave-one-subject-out cross-validation comparing multiple classifiers: Random Forest, SVM, and XGBoost with different class weighting strategies. The ML model is trained on 3 sagittal-plane classes only (TLB, TLF, TUP), while lateral postures (TLL, TLR) are handled separately via geometric rules. The best model is selected based on a composite deployment score.

**4. Real-Time Inference** - The selected model is loaded and used with MediaPipe Pose and a webcam (or iPhone camera via IP) to classify posture in real time. Includes a calibration phase to establish the user's baseline posture before classification begins.

## Setup

### Dependencies

Run the first `pip install` cell in the notebook to install all required packages:

```
!pip install pandas numpy matplotlib seaborn scikit-learn opencv-python mediapipe
```

XGBoost is also used in the training section, so make sure it is available in your environment.

### Running the Notebook

1. Open `SIV_project_final.ipynb` in Jupyter or Google Colab.
2. Run the dependency installation cell at the top.
3. Sections 1-3 (analysis, feature engineering, training) can be run sequentially using `dataset.csv`.
4. Section 4 (real-time inference) requires a webcam or an iPhone connected via IP camera. It also expects three pickle files produced by the training phase: `best_model.pkl`, `label_encoder.pkl`, and `best_feature_columns.pkl`.

## Dataset

The dataset contains 4794 frames from 13 subjects. Each frame includes 3D coordinates (x, y, z) of body landmarks extracted via pose estimation. The class distribution is imbalanced: TLF and TUP together account for about 73% of the samples (imbalance ratio: 4.52).

## Features

### Dataset Analysis Features (13)

The dataset analysis phase extracts 13 geometric features to study posture from all angles:

- **Primary (9)**: head_tilt, head_offset_x, head_forward_distance, head_neck_vertical_angle, shoulder_slope, shoulder_width, head_shoulder_alignment, trunk_upper_lateral_angle, trunk_upper_forward_backward_angle
- **Support (4)**: neck_trunk_angle_anatomical, trunk_lateral_angle, trunk_forward_backward_angle, trunk_depth

### Classification Architecture (Hybrid)

The real-time classification system uses a two-level hybrid approach:

**ML model (sagittal plane)** -- A trained classifier handles the 3 sagittal-plane classes (TLB, TLF, TUP) using 3 features after per-subject z-score normalization: `head_y_ratio` (head vertical displacement projected onto the shoulder-line normal), `head_neck_vertical_angle`, and `trunk_forward_backward_angle`.

**Geometric rules (lateral plane)** -- Lateral postures (TLL, TLR) are detected through threshold-based rules on 4 additional geometric features: `shoulder_slope`, `shoulder_z_asym`, `ear_height_asym`, and `head_lateral_ratio`. The `classify_posture` function combines the ML prediction with lateral scores and overrides the output when the lateral signal exceeds calibrated thresholds.
