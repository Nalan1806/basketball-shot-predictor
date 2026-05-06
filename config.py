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
# Widened range for variable indoor conditions (room lamps, warm/cool light).
BALL_HSV_LOWER = (3, 80, 80)      # Lower bound (Hue, Saturation, Value)
BALL_HSV_UPPER = (28, 255, 255)   # Upper bound (Hue, Saturation, Value)

# Secondary HSV range for brownish/dark basketballs under dim lighting
BALL_HSV_LOWER_2 = (0, 60, 50)    # Catches darker orange / brown tones
BALL_HSV_UPPER_2 = (15, 200, 180) # Restricted to avoid skin false positives

# Enable adaptive HSV calibration via runtime trackbars (press H to toggle)
ENABLE_HSV_TRACKBARS = False

# Morphological operations to clean up the mask
MORPH_KERNEL_SIZE = 5             # Kernel size for erosion/dilation
MORPH_ITERATIONS = 1              # Reduced to 1 — less aggressive cleanup preserves small balls

# Minimum contour area (in pixels²) to consider as a valid ball detection.
# Lowered for close-range indoor room testing where the ball is big on screen.
MIN_CONTOUR_AREA = 300
MAX_CONTOUR_AREA = 80000          # Increased — ball can be very large up close

# Minimum circularity ratio (0.0 - 1.0) to filter non-circular contours.
# A perfect circle has circularity = 1.0
MIN_CIRCULARITY = 0.35            # Slightly relaxed — partial occlusion by hand

# ─────────────────────────────────────────────
# Motion Detection (Stage 2 of hybrid pipeline)
# ─────────────────────────────────────────────
MOG2_HISTORY = 150                # Reduced — adapts faster to room changes
MOG2_VAR_THRESHOLD = 40           # Lowered for better sensitivity indoors
MOG2_LEARNING_RATE = 0.008        # Slightly faster adaptation for indoor use
FRAME_DIFF_THRESHOLD = 20         # Lowered — catches subtler indoor movement

# ─────────────────────────────────────────────
# Detection Confidence Scoring Weights
# ─────────────────────────────────────────────
# These weights control how much each factor contributes to the
# final detection confidence score. They should sum to ~1.0.
W_CIRCULARITY = 0.15              # Shape quality (reduced — hand occlusion hurts this)
W_COLOR = 0.30                    # HSV color match strength (boosted — most reliable indoors)
W_MOTION = 0.15                   # Motion intensity in bounding box
W_AREA_STABILITY = 0.15           # Radius consistency with recent frames
W_TEMPORAL = 0.15                 # Proximity to recent detections
W_HAND_PROXIMITY = 0.10           # NEW: bonus for being near detected hand positions

# Minimum detection confidence to accept a candidate
MIN_DETECTION_CONFIDENCE = 0.20   # Lowered — indoor conditions are harder

# ─────────────────────────────────────────────
# Temporal Filtering
# ─────────────────────────────────────────────
TEMPORAL_WINDOW = 15              # Increased — longer memory helps indoor tracking

# ─────────────────────────────────────────────
# Ball Tracking
# ─────────────────────────────────────────────
MAX_TRACK_POINTS = 50             # Increased — more trail data for prediction
MAX_FRAMES_MISSING = 12           # Increased — ball often briefly occluded by hands
MAX_DISTANCE_JUMP = 250           # Increased — fast movements in close range

# ─────────────────────────────────────────────
# Kalman Filter Parameters
# ─────────────────────────────────────────────
KALMAN_PROCESS_NOISE = 5.0        # Slightly higher — indoor ball movement is more erratic
KALMAN_MEASUREMENT_NOISE = 12.0   # Lowered — trust detection more when it's available
KALMAN_PREDICT_FRAMES = 8         # Increased — predict longer during hand occlusion

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
# YOLO Fallback (Optional — NOT default)
# ─────────────────────────────────────────────
ENABLE_YOLO_FALLBACK = False      # Set True to enable ML fallback
YOLO_MODEL_PATH = "yolov8n.pt"    # Path to YOLOv8-nano weights
YOLO_CONFIDENCE = 0.3             # Min confidence for YOLO detections
YOLO_FALLBACK_THRESHOLD = 0.2     # Classical confidence below this triggers YOLO

# ─────────────────────────────────────────────
# MediaPipe Body Pose Settings
# ─────────────────────────────────────────────
ENABLE_BODY_TRACKING = True       # Master toggle for body/hand tracking
POSE_MODEL_COMPLEXITY = 1         # 0=Lite, 1=Full, 2=Heavy (1 = good balance)
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_VISIBILITY_THRESHOLD = 0.5   # Min visibility to consider a landmark valid

# ─────────────────────────────────────────────
# MediaPipe Hand Tracking Settings
# ─────────────────────────────────────────────
HAND_MODEL_COMPLEXITY = 1         # 0=Lite, 1=Full
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5

# ─────────────────────────────────────────────
# Body-Ball Association
# ─────────────────────────────────────────────
# Max distance (pixels) between ball center and hand for "in-hand" state
BALL_HAND_PROXIMITY_THRESHOLD = 120

# Wrist vertical velocity threshold (pixels/3frames) to detect shot release
# Negative because upward = decreasing Y in pixel coords
RELEASE_VELOCITY_THRESHOLD = 30

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

# Debug-specific colors
COLOR_KALMAN_PREDICTED = (255, 0, 255) # Magenta — Kalman predicted position
COLOR_RAW_DETECTION = (0, 255, 255)    # Yellow — raw (uncorrected) detection
COLOR_CANDIDATE = (255, 100, 0)        # Blue — candidate contours

# Body tracking overlay colors
COLOR_POSE_LANDMARKS = (100, 255, 200)  # Mint — body joints
COLOR_POSE_CONNECTIONS = (80, 180, 140) # Teal — body skeleton lines
COLOR_HAND_LANDMARKS = (255, 180, 50)   # Gold — hand joint dots
COLOR_HAND_CONNECTIONS = (200, 140, 30) # Dark gold — hand bone lines
COLOR_SHOOTING_ARM = (0, 200, 255)      # Orange — highlighted shooting arm
COLOR_ANGLE_ARC = (255, 220, 100)       # Light cyan — joint angle arc
COLOR_BALL_HAND_LINK = (150, 255, 150)  # Light green — ball-to-hand link line

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
