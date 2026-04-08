"""
Configuration module for the Basketball Shot Prediction System.

All tunable parameters are centralized here for easy adjustment
across different environments (laptop, Raspberry Pi, different courts).
"""


# ─────────────────────────────────────────────
# Camera Settings
# ─────────────────────────────────────────────
CAMERA_INDEX = 0                # Webcam device index (0 = default)
FRAME_WIDTH = 1280              # Capture resolution width
FRAME_HEIGHT = 720              # Capture resolution height
FPS_TARGET = 30                 # Target frames per second

# ─────────────────────────────────────────────
# Ball Detection — HSV Color Range
# ─────────────────────────────────────────────
# Standard orange basketball under indoor lighting.
# Adjust these if lighting conditions change.
BALL_HSV_LOWER = (5, 100, 100)    # Lower bound (Hue, Saturation, Value)
BALL_HSV_UPPER = (25, 255, 255)   # Upper bound (Hue, Saturation, Value)

# Morphological operations to clean up the mask
MORPH_KERNEL_SIZE = 5             # Kernel size for erosion/dilation
MORPH_ITERATIONS = 2              # Number of erosion/dilation passes

# Minimum contour area (in pixels²) to consider as a valid ball detection.
# Helps filter out noise. Adjust based on how far the ball is from camera.
MIN_CONTOUR_AREA = 500
MAX_CONTOUR_AREA = 50000

# Minimum circularity ratio (0.0 - 1.0) to filter non-circular contours.
# A perfect circle has circularity = 1.0
MIN_CIRCULARITY = 0.4

# ─────────────────────────────────────────────
# Ball Tracking
# ─────────────────────────────────────────────
MAX_TRACK_POINTS = 40             # Max trajectory points to keep in buffer
MAX_FRAMES_MISSING = 8            # Frames before we consider the ball lost
MAX_DISTANCE_JUMP = 200           # Max pixel distance between consecutive detections

# ─────────────────────────────────────────────
# Trajectory Prediction
# ─────────────────────────────────────────────
POLY_DEGREE = 2                   # Polynomial degree for curve fitting (2 = parabolic)
MIN_POINTS_FOR_PREDICTION = 6     # Minimum tracked points before predicting
PREDICTION_STEPS = 30             # How many steps into the future to predict
PREDICTION_STEP_SIZE = 10         # Pixel step size for X extrapolation
SMOOTHING_WINDOW = 5              # Moving average window for smoothing positions

# ─────────────────────────────────────────────
# Hoop Region (defined as a rectangle in pixel coords)
# ─────────────────────────────────────────────
# These should be calibrated per setup. Set to None to enter
# interactive calibration mode on first run.
HOOP_X = None                     # Top-left X of hoop region
HOOP_Y = None                     # Top-left Y of hoop region
HOOP_WIDTH = 120                  # Width of hoop region box
HOOP_HEIGHT = 80                  # Height of hoop region box

# Intersection tolerance — how close (pixels) the predicted
# trajectory must come to the hoop center to count as a SCORE
INTERSECTION_TOLERANCE = 60

# ─────────────────────────────────────────────
# Visual Overlay Settings
# ─────────────────────────────────────────────
# Colors in BGR format (OpenCV convention)
COLOR_BALL_BBOX = (0, 255, 255)         # Yellow — ball bounding box
COLOR_TRAJECTORY = (255, 200, 0)        # Cyan-ish — past trajectory dots
COLOR_PREDICTION = (0, 165, 255)        # Orange — predicted trajectory curve
COLOR_HOOP = (0, 255, 0)               # Green — hoop region
COLOR_SCORE = (0, 255, 100)            # Green — SCORE label
COLOR_MISS = (0, 50, 255)              # Red — MISS label
COLOR_FPS = (200, 200, 200)            # Light gray — FPS counter
COLOR_INFO = (255, 255, 255)           # White — info text
COLOR_PANEL_BG = (30, 30, 30)          # Dark panel background
COLOR_CONFIDENCE_HIGH = (0, 255, 100)  # Green — high confidence
COLOR_CONFIDENCE_LOW = (0, 100, 255)   # Orange — low confidence

# Font settings
FONT = None  # Will use cv2.FONT_HERSHEY_SIMPLEX (set in code)
FONT_SCALE_LARGE = 1.8
FONT_SCALE_MEDIUM = 0.7
FONT_SCALE_SMALL = 0.55
FONT_THICKNESS = 2

# Overlay element sizes
TRAJECTORY_DOT_RADIUS = 4
PREDICTION_DOT_RADIUS = 3
BBOX_THICKNESS = 2
HOOP_THICKNESS = 3
