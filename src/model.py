# src/model.py
"""
LSTM model definition for position and velocity estimation.
"""

import torch
import torch.nn as nn


class PositionVelocityLSTM(nn.Module):
    """
    A simple LSTM-based regression model that predicts [x, y, vx, vy]
    from a sequence of UWB-based features.
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)      # (batch, seq_len, hidden)
        last_out = lstm_out[:, -1, :]   # (batch, hidden)
        preds = self.fc(last_out)       # (batch, output_size)
        return preds
