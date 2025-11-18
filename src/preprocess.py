# src/preprocess.py
"""
Preprocessing Module

This module transforms raw UWB measurements into sliding-window sequences
suitable for LSTM training.

Main steps:
    - Normalize distance
    - Normalize output targets (x, y, vx, vy)
    - One-hot encode anchor_id
    - Detect constant-velocity segments
    - Build sliding windows of shape (window_size, 8)
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_training_windows(
    df: pd.DataFrame,
    window_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, List[int]]:
    """
    Convert UWB measurement DataFrame into LSTM-ready windowed inputs.

    Args:
        df: DataFrame of UWB measurements.
        window_size: Number of consecutive samples per LSTM input window.

    Returns:
        X_sequences: np.ndarray shaped (num_windows, window_size, 8)
        Y_targets:   np.ndarray shaped (num_windows, 4)
        scaler_distance: fitted scaler for distance
        scaler_output: fitted scaler for x,y,vx,vy
        anchor_ids: sorted list of anchors used
    """

    if "sequence_number" in df.columns:
        df = df.drop(columns=["sequence_number"])

    # ----------------------------------
    # 1) Fit scalers for distance & outputs
    # ----------------------------------
    scaler_distance = MinMaxScaler()
    scaler_output = MinMaxScaler()

    distance_values = df[["distance"]].values
    scaler_distance.fit(distance_values)
    df["distance_norm"] = scaler_distance.transform(distance_values)

    output_values = df[["x", "y", "vx", "vy"]].values
    scaler_output.fit(output_values)
    df[["x_norm", "y_norm", "vx_norm", "vy_norm"]] = scaler_output.transform(output_values)

    # ----------------------------------
    # 2) One-hot encode anchor_id
    # ----------------------------------
    anchor_ids = sorted(df["anchor_id"].unique())
    anchor_to_idx = {aid: i for i, aid in enumerate(anchor_ids)}
    num_anchors = len(anchor_ids)

    anchor_onehot = np.zeros((len(df), num_anchors))
    for i, aid in enumerate(df["anchor_id"]):
        anchor_onehot[i, anchor_to_idx[aid]] = 1.0

    # ----------------------------------
    # 3) Construct feature vectors:
    #    [one-hot anchors (len(anchor_ids)), distance_norm, delta_t]
    # ----------------------------------
    X_features = []
    for i, row in df.iterrows():
        feat_vec = list(anchor_onehot[i])
        feat_vec.append(row["distance_norm"])
        feat_vec.append(row["delta_t"])
        X_features.append(feat_vec)
    X_features = np.array(X_features)

    Y_targets = df[["x_norm", "y_norm", "vx_norm", "vy_norm"]].to_numpy()

    # ----------------------------------
    # 4) Detect constant-velocity segments
    # ----------------------------------
    segment_indices = [0]
    for i in range(1, len(df)):
        prev_vx, prev_vy = df.loc[i - 1, ["vx_norm", "vy_norm"]]
        curr_vx, curr_vy = df.loc[i, ["vx_norm", "vy_norm"]]

        # Significant velocity change -> new segment
        if abs(curr_vx - prev_vx) > 1e-6 or abs(curr_vy - prev_vy) > 1e-6:
            segment_indices.append(i)
    segment_indices.append(len(df))

    # ----------------------------------
    # 5) Build sliding windows
    # ----------------------------------
    X_sequences = []
    Y_sequence_targets = []

    for s in range(len(segment_indices) - 1):
        start = segment_indices[s]
        end = segment_indices[s + 1]

        X_seg = X_features[start:end]
        Y_seg = Y_targets[start:end]
        seg_len = len(X_seg)

        if seg_len < window_size + 1:
            continue

        # window_size inputs -> next-step target
        for i in range(seg_len - window_size):
            X_sequences.append(X_seg[i : i + window_size])
            Y_sequence_targets.append(Y_seg[i + window_size])

    X_sequences = np.array(X_sequences)
    Y_sequence_targets = np.array(Y_sequence_targets)

    print("Number of windows:", X_sequences.shape[0])
    return X_sequences, Y_sequence_targets, scaler_distance, scaler_output, anchor_ids
