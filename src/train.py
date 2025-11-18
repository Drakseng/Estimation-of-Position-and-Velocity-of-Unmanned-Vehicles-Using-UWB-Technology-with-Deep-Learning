# src/train.py
"""
Training utilities for the LSTM model:
- model training loop
- saving and loading artifacts (model + scalers + anchor_ids)
"""

from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from .model import PositionVelocityLSTM


def train_model(
    X_sequences: np.ndarray,
    Y_targets: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> PositionVelocityLSTM:
    """
    Train the LSTM model on windowed UWB data.

    Args:
        X_sequences: input windows, shape (N, T, F)
        Y_targets:  target vectors, shape (N, 4)
        epochs:     number of training epochs
        batch_size: mini-batch size
        lr:         learning rate
        device:     torch.device (CPU/GPU), auto-detected if None

    Returns:
        Trained PositionVelocityLSTM model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PositionVelocityLSTM(
        input_size=X_sequences.shape[-1],
        output_size=Y_targets.shape[-1],
    ).to(device)

    X_tensors = torch.tensor(X_sequences, dtype=torch.float32)
    Y_tensors = torch.tensor(Y_targets, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensors, Y_tensors)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_Y in loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    return model


def save_artifacts(
    model: PositionVelocityLSTM,
    scaler_distance: MinMaxScaler,
    scaler_output: MinMaxScaler,
    anchor_ids: List[int],
    model_path: str = "model.pt",
    scaler_x_path: str = "scaler_x.pkl",
    scaler_y_path: str = "scaler_y.pkl",
    anchor_ids_path: str = "anchor_ids.pkl",
) -> None:
    """
    Save model parameters, scalers and anchor ID mapping to disk.
    """
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_distance, scaler_x_path)
    joblib.dump(scaler_output, scaler_y_path)
    joblib.dump(anchor_ids, anchor_ids_path)
    print("Model and scalers saved.")


def load_artifacts(
    model_path: str = "model.pt",
    scaler_x_path: str = "scaler_x.pkl",
    scaler_y_path: str = "scaler_y.pkl",
    anchor_ids_path: str = "anchor_ids.pkl",
    device: Optional[torch.device] = None,
) -> Tuple[PositionVelocityLSTM, MinMaxScaler, MinMaxScaler, List[int]]:
    """
    Load model, scalers and anchor list from disk.

    Args:
        model_path:      path to saved model weights
        scaler_x_path:   path to distance scaler
        scaler_y_path:   path to output scaler
        anchor_ids_path: path to saved anchor_ids list
        device:          target device

    Returns:
        (model, scaler_distance, scaler_output, anchor_ids)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PositionVelocityLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scaler_distance = joblib.load(scaler_x_path)
    scaler_output = joblib.load(scaler_y_path)
    anchor_ids: List[int] = joblib.load(anchor_ids_path)

    return model, scaler_distance, scaler_output, anchor_ids
