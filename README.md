<img width="666" height="386" alt="image" src="https://github.com/user-attachments/assets/cc5f1aca-adfe-4479-9207-965d1892b11f" /># 🚀 Estimation-of-Position-and-Velocity-of-Unmanned-Vehicles-Using-UWB-Technology-with-Deep-Learning
Deep learning based position and velocity estimation for unmanned vehicles using UWB distance measurements, featuring RNN modeling, synthetic data simulation, and Kalman Filter comparison.

Estimation of Position and Velocity of Unmanned Vehicles Using UWB Technology with Deep Learning

This project explores a deep learning–based localization system designed to estimate the position and velocity of an unmanned vehicle using Ultra-Wideband (UWB) distance measurements.
The system combines:

A synthetic UWB simulation environment

Sliding-window time-series preprocessing

A multi-layer LSTM neural network

Optional Kalman Filter baseline comparison

Trajectory and velocity-field visualizations

The aim is to present a robust, high-resolution indoor localization method suitable for environments where GPS is unreliable or unavailable.

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


A simulated unmanned vehicle moves with:

-Random speed ∈ [0.1, 1.5] m/s
-Random heading (0–2π)
-Perfectly elastic boundary reflections
-Continuous velocity segments

At each burst, 2–6 anchors provide UWB range measurements with a timestamp offset.

## 3 Dataset Construction (Windowing Approach)

For training:

Each sample consists of 10 consecutive measurements

Each measurement includes:

[one-hot anchor_id (6 dims), normalized distance, delta_t]


Total input dimension = 8

Output = next state:

[x, y, vx, vy]

Final shapes:

Input windows: (77,507 × 10 × 8)

Targets: (77,507 × 4)

This makes the model learn time dependencies in the motion.


## 4. Model Architecture 

-2-layer LSTM
-Hidden size: 64
-Optimizer: Adam (1e-3)
-Loss: MSE

Training for 20 epochs reduces loss from 0.039 → 0.009.
