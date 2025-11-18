#!/usr/bin/env python
# coding: utf-8
"""
Main Pipeline Script

This file orchestrates the full workflow:

1) (Optional) UWB simulation
2) Preprocessing: scaling + sliding-window generation
3) LSTM training
4) Saving model + scalers (artifacts)
5) Inference and trajectory visualization on test CSV files

Note:
Real measurement data belongs to a defense industry project and cannot be shared.
You may provide your own UWB CSV files for inference.
"""

import os
import torch

from src.simulate import simulate_uwb_data
from src.preprocess import build_training_windows
from src.train import train_model, save_artifacts, load_artifacts
from src.inference import predict_trajectory_for_file
from src.visualize import plot_trajectory_and_velocity


def main():
    # Choose GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==============================================================
    # 1) OPTIONAL: RUN SIMULATION + TRAINING FROM SCRATCH
    # ==============================================================

    print("Simulating UWB data...")
    df_sim = simulate_uwb_data(max_bursts=20000)
    df_sim.to_csv("uwb_data.csv", index=False, float_format="%.3f")
    print("Saved synthetic UWB data to uwb_data.csv")

    print("Building sliding-window training set...")
    X_sequences, Y_targets, scaler_distance, scaler_output, anchor_ids = build_training_windows(df_sim)

    print("Training LSTM model...")
    model = train_model(X_sequences, Y_targets, epochs=20, batch_size=64, lr=1e-3, device=device)

    print("Saving model and scalers...")
    save_artifacts(model, scaler_distance, scaler_output, anchor_ids)

    # ==============================================================
    # 2) EXAMPLE INFERENCE ON TEST FILES
    # ==============================================================
    # If you want to skip training and use an existing model, comment out
    # the simulation+training above and uncomment the line below:
    #
    # model, scaler_distance, scaler_output, anchor_ids = load_artifacts(device=device)

    test_files = [
        ("test_edilecek.csv", "predicted_trajectory.csv",      (0, 28), (0, 28)),
        ("test_edilecek_2_icin.csv", "predicted_trajectory_2.csv", (0, 28), (0, 15)),
        ("test_edilecek_4_icin.csv", "predicted_trajectory_4.csv", (0, 28), (0, 28)),
    ]

    # Process each test file (if it exists)
    for input_csv, output_csv, x_lim, y_lim in test_files:
        if os.path.exists(input_csv):
            print(f"Running inference for {input_csv}...")
            pred_df = predict_trajectory_for_file(
                csv_path=input_csv,
                output_path=output_csv,
                model=model,
                scaler_distance=scaler_distance,
                scaler_output=scaler_output,
                anchor_ids=anchor_ids,
                device=device,
            )
            plot_trajectory_and_velocity(
                pred_df,
                title_prefix=f"Predicted Trajectory ({input_csv})",
                xlim=x_lim,
                ylim=y_lim,
            )
        else:
            print(f"Test file not found, skipping: {input_csv}")


if __name__ == "__main__":
    main()

