# src/inference.py
"""
Inference Module

This module loads a trained LSTM model and runs inference on new
UWB measurement files to predict [x, y, vx, vy].
"""

from typing import List, Optional

import numpy as np
import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from .model import PositionVelocityLSTM


def predict_trajectory_for_file(
    csv_path: str,
    output_path: str,
    model: PositionVelocityLSTM,
    scaler_distance: MinMaxScaler,
    scaler_output: MinMaxScaler,
    anchor_ids: List[int],
    window_size: int = 10,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Predict trajectory for a new UWB measurement file.

    Expected CSV columns:
        anchor_id, distance, delta_t

    Processing Steps:
        1. One-hot encode anchor_id
        2. Normalize distance using training scaler
        3. Build feature vectors: [one-hot, distance_norm, delta_t]
        4. Create sliding windows (window_size)
        5. Run inference with LSTM
        6. Inverse-transform predictions to real-world scale
        7. Save output CSV

    Returns:
        DataFrame with columns:
            pred_x, pred_y, pred_vx, pred_vy
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df_test = pd.read_csv(csv_path)

    if "sequence_number" in df_test.columns:
        df_test = df_test.drop(columns=["sequence_number"])

    # Map anchors from training to one-hot indices
    anchor_to_idx = {aid: i for i, aid in enumerate(anchor_ids)}
    num_anchors = len(anchor_ids)

    test_onehot = np.zeros((len(df_test), num_anchors))
    for i, aid in enumerate(df_test["anchor_id"]):
        if aid in anchor_to_idx:
            test_onehot[i, anchor_to_idx[aid]] = 1.0

    # Normalize distance using training scaler
    distance_vals = df_test[["distance"]].values
    df_test["distance_norm"] = scaler_distance.transform(distance_vals)

    # Build feature vectors
    X_test_features = []
    for i, row in df_test.iterrows():
        feat_vec = list(test_onehot[i])
        feat_vec.append(row["distance_norm"])
        feat_vec.append(row["delta_t"])
        X_test_features.append(feat_vec)
    X_test_features = np.array(X_test_features)

    # Build sliding windows
    X_test_windows = []
    for i in range(len(X_test_features) - window_size):
        X_test_windows.append(X_test_features[i : i + window_size])

    if not X_test_windows:
        raise ValueError("Not enough samples to build windows.")

    X_test_tensor = torch.tensor(np.array(X_test_windows), dtype=torch.float32).to(device)

    # Run model
    with torch.no_grad():
        pred_norm = model(X_test_tensor).cpu().numpy()

    # Convert normalized predictions back to physical values
    pred_phys = scaler_output.inverse_transform(pred_norm)

    pred_df = pd.DataFrame(pred_phys, columns=["pred_x", "pred_y", "pred_vx", "pred_vy"])
    pred_df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    return pred_df
