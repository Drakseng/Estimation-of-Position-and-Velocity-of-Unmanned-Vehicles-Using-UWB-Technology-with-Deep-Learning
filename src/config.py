# src/config.py
"""
Configuration for the UWB simulation environment.
"""

# Area boundaries (meters)
X_MIN, X_MAX = 0.0, 28.0
Y_MIN, Y_MAX = 0.0, 14.0

# Anchor positions (ID: (x, y) in meters)
ANCHORS = {
    1: (0.0,  0.0),
    2: (0.0, 14.0),
    3: (14.0, 14.0),
    4: (28.0, 14.0),
    5: (28.0,  0.0),
    6: (14.0,  0.0),
}
