# src/simulate.py
"""
UWB Simulation Module

This module simulates distance measurements captured by fixed UWB anchors
tracking a moving object in a 2D bounded environment.

Each record corresponds to a single UWB measurement, and includes:
    - segment_id : ID of the velocity segment during which the object moved
    - sequence_id: burst ID (group of measurements taken at the same time)
    - delta_t    : time difference from the previous measurement
    - anchor_id  : ID of the anchor making the measurement
    - distance   : Euclidean distance between anchor and object
    - x, y       : true position of the object at measurement time
    - vx, vy     : true velocity components at measurement time
"""

import math
import random
from typing import List

import pandas as pd

from .config import X_MIN, X_MAX, Y_MIN, Y_MAX, ANCHORS


def compute_time_to_boundary(x: float, y: float, vx: float, vy: float) -> float:
    """
    Compute the time until the moving object (x, y) hits any boundary
    of the defined rectangular area, given velocity components (vx, vy).

    Returns:
        The earliest collision time (seconds).
        If no collision will occur, returns float('inf').
    """
    times: List[float] = []

    if vx > 0:
        times.append((X_MAX - x) / vx)
    elif vx < 0:
        times.append((X_MIN - x) / vx)

    if vy > 0:
        times.append((Y_MAX - y) / vy)
    elif vy < 0:
        times.append((Y_MIN - y) / vy)

    times = [t for t in times if t >= 0]
    return min(times) if times else float("inf")


def simulate_uwb_data(max_bursts: int = 20000) -> pd.DataFrame:
    """
    Generate a synthetic UWB measurement dataset.

    In each burst, 2–6 anchors record the distance to the moving object,
    with very small measurement offsets (milliseconds).

    Args:
        max_bursts: Number of bursts to simulate.

    Returns:
        DataFrame with synthetic UWB measurement records.
    """
    data_records = []

    # Initialize object with random position and velocity
    current_time = 0.0
    current_x = random.uniform(X_MIN, X_MAX)
    current_y = random.uniform(Y_MIN, Y_MAX)
    speed = random.uniform(0.1, 1.5)
    direction = random.uniform(0, 2 * math.pi)
    current_vx = speed * math.cos(direction)
    current_vy = speed * math.sin(direction)

    segment_id = 1
    segment_end_time = current_time + compute_time_to_boundary(
        current_x, current_y, current_vx, current_vy
    )

    sequence_id = 0
    last_measure_time = None

    for _ in range(max_bursts):

        # Determine the next burst start time
        if last_measure_time is None:
            burst_start = current_time
        else:
            gap = random.uniform(0.1, 0.6)
            burst_start = last_measure_time + gap

        # Process boundary collisions before the burst starts
        while burst_start > segment_end_time:
            dt_to_boundary = segment_end_time - current_time

            # Move to boundary point
            current_x += current_vx * dt_to_boundary
            current_y += current_vy * dt_to_boundary
            current_x = max(X_MIN, min(X_MAX, current_x))
            current_y = max(Y_MIN, min(Y_MAX, current_y))
            current_time = segment_end_time

            # Create a new velocity segment at boundary
            segment_id += 1
            speed = random.uniform(0.1, 1.5)
            direction = random.uniform(0, 2 * math.pi)
            current_vx = speed * math.cos(direction)
            current_vy = speed * math.sin(direction)

            segment_end_time = current_time + compute_time_to_boundary(
                current_x, current_y, current_vx, current_vy
            )

        # Move from previous measurement time to burst start
        dt_to_burst = burst_start - current_time
        current_x += current_vx * dt_to_burst
        current_y += current_vy * dt_to_burst
        current_time = burst_start

        sequence_id += 1

        # Select random anchors for the burst
        anchor_count = random.randint(2, 6)
        anchor_ids = random.sample(list(ANCHORS.keys()), anchor_count)

        # Small offsets inside the same burst
        offsets = sorted(random.uniform(0.001, 0.01) for _ in range(anchor_count))

        for offset, anchor_id in zip(offsets, anchor_ids):
            measure_time = burst_start + offset

            # Handle boundary collisions before measurement time
            while measure_time > segment_end_time:
                dt_to_boundary = segment_end_time - current_time
                current_x += current_vx * dt_to_boundary
                current_y += current_vy * dt_to_boundary

                current_x = max(X_MIN, min(X_MAX, current_x))
                current_y = max(Y_MIN, min(Y_MAX, current_y))
                current_time = segment_end_time

                segment_id += 1
                speed = random.uniform(0.1, 1.5)
                direction = random.uniform(0, 2 * math.pi)
                current_vx = speed * math.cos(direction)
                current_vy = speed * math.sin(direction)

                segment_end_time = current_time + compute_time_to_boundary(
                    current_x, current_y, current_vx, current_vy
                )

            # Move to measurement time
            dt_meas = measure_time - current_time
            current_x += current_vx * dt_meas
            current_y += current_vy * dt_meas
            current_time = measure_time

            # Anchor-object distance
            ax, ay = ANCHORS[anchor_id]
            distance = math.hypot(current_x - ax, current_y - ay)

            # Time since last measurement
            delta_t = (current_time - last_measure_time) if last_measure_time is not None else 0.0
            last_measure_time = current_time

            data_records.append(
                {
                    "segment_id": segment_id,
                    "sequence_id": sequence_id,
                    "delta_t": delta_t,
                    "anchor_id": anchor_id,
                    "distance": distance,
                    "x": current_x,
                    "y": current_y,
                    "vx": current_vx,
                    "vy": current_vy,
                }
            )

    return pd.DataFrame(data_records)
