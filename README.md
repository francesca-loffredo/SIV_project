# SIV Project - Upper Body Posture Analysis

A posture classification system that detects upper body posture from webcam input using MediaPipe pose estimation and machine learning. The system classifies five posture types: lean backward (TLB), lean forward (TLF), lean left (TLL), lean right (TLR), and upright (TUP).

## Pipeline Overview

The notebook is organized into four main sections:

**1. Dataset Analysis** - Exploration of the raw landmark data: class distribution, landmark quality checks, geometric consistency verification.

**2. Feature Engineering** - Extraction of 13 geometric features from body landmarks (head, shoulders, hips). Features are divided into 9 primary features (head and shoulder based) and 4 support features (trunk based). Distances and offsets are normalized by shoulder width. A per-subject z-score normalization with a standard deviation floor is applied before training.

**3. ML Model Training** - Leave-one-subject-out cross-validation comparing multiple classifiers: Random Forest, SVM, and XGBoost with different class weighting strategies. Lateral postures (TLL, TLR) are merged into a single lateral class before training. The best model is selected based on a composite deployment score.

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

The final feature set includes 13 geometric variables:

- **Primary**: head_tilt, head_offset_x, head_forward_distance, head_neck_vertical_angle, shoulder_slope, shoulder_width, head_shoulder_alignment, trunk_upper_lateral_angle, trunk_upper_forward_backward_angle
- **Support**: neck_trunk_angle_anatomical, trunk_lateral_angle, trunk_forward_backward_angle, trunk_depth

For the ML training phase, a reduced set of 3 features is used after z-score normalization: `head_y_ratio`, `head_neck_vertical_angle`, and `trunk_forward_backward_angle`.
