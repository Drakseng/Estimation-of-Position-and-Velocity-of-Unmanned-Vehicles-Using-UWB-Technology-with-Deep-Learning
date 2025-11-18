# 🚀 Estimation of Position and Velocity of-Unmanned Vehicles Using UWB Technology with Deep Learning
Deep learning based position and velocity estimation for unmanned vehicles using UWB distance measurements, featuring RNN modeling, synthetic data simulation, and Kalman Filter comparison.

Estimation of Position and Velocity of Unmanned Vehicles Using UWB Technology with Deep Learning

This project explores a deep learning–based localization system designed to estimate the position and velocity of an unmanned vehicle using Ultra-Wideband (UWB) distance measurements.
The system combines:

-A synthetic UWB simulation environment
-Sliding-window time-series preprocessing
-A multi-layer LSTM neural network
-Optional Kalman Filter baseline comparison
-Trajectory and velocity-field visualizations

The aim is to present a robust, high-resolution indoor localization method suitable for environments where GPS is unreliable or unavailable.

#### Table of Contents:

1. Project Motivation
2. UWB Simulation Environment
3. Dataset Construction (Windowing Approach)
4. Model Architecture
5. Results
    • Scenario I (Real Data – “I” Route)
    • Scenario U (Real Data – “U” Route)
6. Usage (How to Run)
7. Data Availability

## 1. Project Motivation

Localization is a critical capability for unmanned systems operating indoors or GPS-denied environments.
Traditional methods such as:

-Kalman Filter
-IMU-only dead reckoning
-GPS

face limitations due to drift, noise, low sampling rates, or unavailability.

UWB technology, with its centimeter-level accuracy and low-power pulses, provides an excellent alternative.

This project demonstrates that a deep learning approach (RNN) can learn the nonlinear spatial–temporal relationships between sequential UWB measurements and accurately reconstruct 2D trajectory + velocity.

## 2. UWB Simulation Environment

The environment is a 28m × 14m indoor arena, surrounded by 6 fixed UWB anchors:

Anchor ID	Coordinates (m)
1	(0, 0)
2	(0, 14)
3	(14, 14)
4	(28, 14)
5	(28, 0)
6	(14, 0)


<img width="666" height="386" alt="image (6)" src="https://github.com/user-attachments/assets/2633ea2f-c5b5-4350-91a6-dd2f354e2c62" />
Figure 1: Visualization of how distance data collected using UWB modules is gathered


A simulated unmanned vehicle moves with:

-Random speed ∈ [0.1, 1.5] m/s

-Random heading (0–2π)

-Perfectly elastic boundary reflections

-Continuous velocity segments

-At each burst, 2–6 anchors provide UWB range measurements with a timestamp offset.

## 3 Dataset Construction (Windowing Approach)

For training:

-Each measurement includes: [one-hot anchor_id (6 dims), normalized distance, delta_t]
-Total input dimension = 8
-Output = next state: [x, y, vx, vy]

Final shapes:

-Input windows: (77,507 × 10 × 8)
-Targets: (77,507 × 4)

This makes the model learn time dependencies in the motion.


## 4. Model Architecture 

-2-layer LSTM
-Hidden size: 64
-Optimizer: Adam (1e-3)
-Loss: MSE

Training for 20 epochs reduces loss from 0.039 → 0.009.


## 5. Results
### Scenario I (Real Data – “I” Route)

In the first scenario, real UWB data was collected along an **I-shaped route**. The model, trained with synthetic data and validated on this dataset, successfully predicted the vehicle’s trajectory with high consistency. Despite the presence of sensor noise and missing measurements, preprocessing steps such as normalization and Kalman filter imputation ensured smooth and accurate estimations. The predicted positions closely followed the true I-shaped path, demonstrating the model’s ability to generalize from synthetic training to real-world testing.

<img width="1045" height="737" alt="notion_1" src="https://github.com/user-attachments/assets/e48d9413-920f-4653-94c7-c30f36d5d8da" />
Figure 2: Scaled Position Estimation for the “I” Trajectory


<img width="575" height="506" alt="notion_2" src="https://github.com/user-attachments/assets/9d123b41-27dd-4ed8-bdc5-253a30ee6ab5" />

Figure 3: Kalman Filter Output for the “I” Trajectory

### Scenario U (Real Data – “U” Route)

In the second scenario, real-world measurements were gathered along a **U-shaped route**. The model again achieved reliable predictions, capturing both the linear and curved sections of the trajectory. Robust preprocessing and sequential learning allowed the system to handle noise and incomplete data effectively. Compared to classical approaches, the model showed superior accuracy in reconstructing the U-shaped path, confirming its robustness in varying movement patterns and environments.

<img width="1011" height="722" alt="notion_3" src="https://github.com/user-attachments/assets/f93527f4-1c9b-4b3f-a65c-8ba5cadc8781" />
Figure 4: Scaled Position Estimation for the “U” Trajectory

<img width="575" height="507" alt="notion_4" src="https://github.com/user-attachments/assets/79b391ef-3648-4576-9829-429606de66db" />

Figure 5: Kalman Filter Output for the “U” Trajectory

## 6. Usage (How to Run)

Since the dataset used in this project is private and cannot be shared, the following steps describe how to run the code structure and how to perform inference using your own UWB measurement data.

This project uses the following Python libraries:

-pandas –> data processing and table operations
-numpy –> numerical computations
-scikit-learn –> normalization (MinMaxScaler)
-joblib –> saving/loading scalers
-torch (PyTorch) –> LSTM model, training, inference
-matplotlib –> trajectory & velocity visualization

All other imports (math, random) are Python built-in modules and require no installation.

#### 1. Running the full pipeline

The main script includes the complete workflow:

-UWB simulation
-Sliding-window dataset preparation
-Feature normalization
-LSTM model training
-Model and scaler saving
-Inference + visualization modules

You can execute the entire pipeline with: python bitti_mi_dayi_tumverisetleri_denendi.py

Note: The original dataset is not included. To reproduce training, you must provide your own UWB file or activate the simulation block inside the script.

#### 2. Using your own UWB measurement file for inference

Prepare a CSV file containing at least: anchor_id, distance, delta_t

Inside the script, update the test file path: df_test = pd.read_csv("your_file.csv")

Then run: python bitti_mi_dayi_tumverisetleri_denendi.py

After inference, predictions will be saved to: predicted_trajectory.csv

with the following columns: pred_x, pred_y, pred_vx, pred_vy

#### 3. Visualizing predictions

You can use the plotting functions inside the script to visualize the predicted motion:
plt.plot(df['pred_x'], df['pred_y'])
plt.quiver(df['pred_x'], df['pred_y'], df['pred_vx'], df['pred_vy'])

## 7. Data Availability (Important Notice)

The measurement data used for training and evaluation was obtained as part of a project conducted under a Defense Industry Research Program. In accordance with institutional confidentiality rules, the dataset cannot be distributed, published, or stored in public repositories.



