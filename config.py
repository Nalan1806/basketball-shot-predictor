"""
Configuration module for ShotIQ — Basketball Shot Prediction System.

Simplified for demo reliability. All tunable parameters centralized here.
"""


# ─────────────────────────────────────────────
# Input/Output settings are now handled via command line arguments.
# Physics and drawing parameters will scale dynamically based on video height.


# ─────────────────────────────────────────────
# YOLO Detection
# ─────────────────────────────────────────────
YOLO_MODEL_PATH = "yolov8n.pt"  # Path to YOLOv8-nano weights
YOLO_CONFIDENCE = 0.10          # Min confidence for YOLO detections (lowered for dim lighting)
YOLO_SPORTS_BALL_CLASS = 32     # COCO class ID for sports ball
YOLO_SKIP_FRAMES = 1            # Offline mode: process EVERY frame for maximum cinematic smoothness


# ─────────────────────────────────────────────
# Detection Stability
# ─────────────────────────────────────────────
MAX_JUMP_PX = 50                # Max pixel jump from last known position (scaled for 480p)
                                # (reduced from 80 to prevent false snapping)
OUTLIER_FRAMES_REQUIRED = 2     # Consecutive detections required after a big jump
                                # to accept a new tracking target


# ─────────────────────────────────────────────
# Kalman Tracking
# ─────────────────────────────────────────────
MAX_TRACK_POINTS = 50           # Max tracked position history
MIN_POINTS_FOR_PREDICTION = 8   # Minimum tracked points before predicting
MAX_FRAMES_MISSING = 30         # Max frames to coast before reset (increased to handle dropped detections in dim lighting)

KALMAN_PROCESS_NOISE = 0.1      # Process noise (lower = strongly favor smooth constant velocity)
KALMAN_MEASUREMENT_NOISE = 30.0 # Measurement noise (higher = heavily smooths YOLO jitter for cinematic look)


# ─────────────────────────────────────────────
# Trajectory Prediction
# ─────────────────────────────────────────────
PREDICTION_STEPS = 60           # How many steps to project into the future
GRAVITY_PX_PER_S2 = 500.0       # Base gravity for 720p (will be auto-scaled by video resolution)

SMOOTHING_WINDOW = 5            # Moving average window for smoothing positions


# ─────────────────────────────────────────────
# Hoop Region (pixel coordinates)
# ─────────────────────────────────────────────
# Static defaults — set these for your setup, or press C to calibrate.
# Set HOOP_X to None to force calibration on first run.
HOOP_X = None                   # Top-left X of hoop region
HOOP_Y = None                   # Top-left Y of hoop region
HOOP_WIDTH = 120                # Base width for 720p (auto-scaled)
HOOP_HEIGHT = 80                # Base height for 720p (auto-scaled)

# Intersection tolerance — how close (pixels) the predicted
# trajectory must come to the hoop center to count as a SCORE
INTERSECTION_TOLERANCE = 60     # Base tolerance for 720p (auto-scaled)


# ─────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────
DT_MIN = 0.005                  # Minimum dt (200 FPS cap) — prevents div-by-zero
DT_MAX = 0.10                   # Maximum dt (10 FPS floor) — prevents physics explosion


# ─────────────────────────────────────────────
# Visual Overlay Settings
# ─────────────────────────────────────────────
# Colors in BGR format (OpenCV convention)
COLOR_BALL_CIRCLE = (0, 220, 0)         # Green — ball indicator
COLOR_TRAJECTORY = (0, 0, 255)          # Red — past trajectory arc
COLOR_PREDICTION = (255, 100, 0)        # Blue — predicted trajectory
COLOR_HOOP = (0, 200, 0)               # Green — hoop region
COLOR_HOOP_SCORE = (0, 255, 80)        # Bright green — hoop on SCORE
COLOR_SCORE = (0, 255, 100)            # Green — SCORE label
COLOR_MISS = (0, 50, 255)              # Red — MISS label
COLOR_FPS = (180, 180, 180)            # Light gray — FPS counter
COLOR_INFO = (255, 255, 255)           # White — info text
COLOR_PANEL_BG = (30, 30, 30)          # Dark panel background

# Font settings
FONT_SCALE_LARGE = 1.8
FONT_SCALE_MEDIUM = 0.7
FONT_SCALE_SMALL = 0.55
FONT_THICKNESS = 2

# Overlay element sizes
TRAJECTORY_DOT_RADIUS = 4
PREDICTION_DOT_RADIUS = 3
BBOX_THICKNESS = 2
HOOP_THICKNESS = 3
