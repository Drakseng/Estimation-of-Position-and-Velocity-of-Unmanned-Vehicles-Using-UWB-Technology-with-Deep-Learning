# src/visualize.py
"""
Visualization utilities for predicted trajectories and velocity vectors.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectory_and_velocity(
    df: pd.DataFrame,
    title_prefix: str = "Predicted Trajectory",
    xlim: Tuple[float, float] = (0, 28),
    ylim: Tuple[float, float] = (0, 28),
) -> None:
    """
    Plot predicted positions and velocity vectors.

    Args:
        df: DataFrame with columns [pred_x, pred_y, pred_vx, pred_vy]
        title_prefix: base title for figures
        xlim: x-axis limits
        ylim: y-axis limits
    """
    # Trajectory plot
    plt.figure(figsize=(8, 6))
    plt.plot(df["pred_x"], df["pred_y"], marker="o")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title(title_prefix)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Velocity vector field
    plt.figure(figsize=(8, 6))
    plt.quiver(
        df["pred_x"],
        df["pred_y"],
        df["pred_vx"],
        df["pred_vy"],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
    )
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title(f"{title_prefix} with Velocity Vectors")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
