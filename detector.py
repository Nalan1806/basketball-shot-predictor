"""
YOLOv8 Basketball Detector — ShotIQ

Real-time detection using YOLOv8n (nano) model:
  - Class 32 (sports ball) detection only
  - Confidence threshold: 0.35
  - Proximity gate: 400px max jump from last known position
  - Selects highest-confidence detection if multiple balls present
  - Falls back to closest-to-last-position if tracking was active

Model downloads automatically on first run (ultralytics handles this).
Very fast inference: ~30 FPS on CPU with 1280×720 input.
"""

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

# ─────────────────────────────────────────────
# YOLO Parameters
# ─────────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8n.pt"  # nano model (fastest)
YOLO_CONF_THRESH = 0.35  # confidence threshold for detections
YOLO_SPORTS_BALL_CLASS = 32  # COCO class ID for sports ball

# ─────────────────────────────────────────────
# Proximity Gate
# ─────────────────────────────────────────────
MAX_JUMP_PX = 400  # max pixel jump from last known position


class BallDetector:
    """
    YOLOv8-based basketball detector.

    Features:
      - Uses pre-trained YOLOv8n nano model
      - Filters to COCO class 32 (sports ball) only
      - Confidence threshold: 0.35
      - Proximity gate: 400px max jump
      - Selects best detection (highest confidence, or closest to last position)
      - Automatically downloads model on first run
    """

    def __init__(self):
        """Initialize YOLO detector and load model."""
        self._last_confidence = 0.0
        self._last_known_pos = None  # (px, py) for proximity gate
        
        # Load YOLOv8n model (downloads from Ultralytics hub on first run)
        try:
            self.model = YOLO(YOLO_MODEL_NAME)
            print(f"  YOLOv8 model '{YOLO_MODEL_NAME}' loaded successfully")
        except Exception as e:
            print(f"  ERROR loading YOLOv8 model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Detect basketball using YOLOv8.

        Process:
          1. Run YOLOv8 inference on frame
          2. Filter detections to COCO class 32 (sports ball) only
          3. Keep detections with confidence >= 0.35
          4. Apply proximity gate: max 400px from last known position
          5. Select best detection (highest confidence if no tracking history,
             else closest to last known position for smooth tracking)

        Args:
            frame: BGR image from OpenCV

        Returns:
            tuple: (detection_dict or None, confidence: float)
              - detection_dict keys: cx, cy, x, y, w, h, radius
              - confidence: float 0–1 (YOLO confidence score)
        """
        if self.model is None:
            self._last_confidence = 0.0
            return None, 0.0

        try:
            # Run YOLOv8 inference
            results = self.model(frame, conf=YOLO_CONF_THRESH, verbose=False)
            result = results[0]

            # Extract detections as boxes object
            boxes = result.boxes
            
            # Filter to sports ball class (class 32)
            sports_ball_detections = []
            
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if class_id != YOLO_SPORTS_BALL_CLASS:
                    continue
                
                if conf < YOLO_CONF_THRESH:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center and dimensions
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                
                # Estimate radius from bounding box
                radius = max(w, h) // 2
                
                sports_ball_detections.append({
                    "cx": cx,
                    "cy": cy,
                    "x": x1,
                    "y": y1,
                    "w": w,
                    "h": h,
                    "radius": radius,
                    "conf": conf,
                })
            
            # No sports ball detections found
            if not sports_ball_detections:
                self._last_confidence = 0.0
                return None, 0.0
            
            # Apply proximity gate filtering
            candidates = sports_ball_detections
            
            if self._last_known_pos is not None:
                lx, ly = self._last_known_pos
                nearby = [c for c in candidates
                          if np.hypot(c["cx"] - lx, c["cy"] - ly) <= MAX_JUMP_PX]
                # If we have nearby candidates, use only them
                if nearby:
                    candidates = nearby
                # Otherwise, accept all (first re-appearance after occlusion)
            
            # Select best detection
            # Priority 1: if tracking active, closest to last known position (smooth)
            # Priority 2: highest confidence (first detection or after long gap)
            if self._last_known_pos is not None and len(candidates) > 0:
                lx, ly = self._last_known_pos
                best = min(candidates,
                           key=lambda c: np.hypot(c["cx"] - lx, c["cy"] - ly))
            else:
                best = max(candidates, key=lambda c: c["conf"])
            
            confidence = float(best["conf"])
            self._last_confidence = confidence
            self._last_known_pos = (best["cx"], best["cy"])
            
            # Return detection dict (remove YOLO-specific 'conf' key)
            detection = {
                "cx": best["cx"],
                "cy": best["cy"],
                "x": best["x"],
                "y": best["y"],
                "w": best["w"],
                "h": best["h"],
                "radius": best["radius"],
            }
            
            return detection, confidence
            
        except Exception as e:
            print(f"  YOLO inference error: {e}")
            self._last_confidence = 0.0
            return None, 0.0

    # ──────────────────────────────────────────────────────────────────
    def update_last_position(self, px, py):
        """
        Externally update the proximity gate reference position.

        Called by tracker after EKF update to keep proximity gate in sync
        with the filtered tracking position.

        Args:
            px, py: pixel coordinates of the updated ball position.
        """
        self._last_known_pos = (px, py)

    def clear_last_position(self):
        """
        Reset the proximity gate reference position.

        Called after tracker reset to allow new detections from any frame
        location on next detection pass.
        """
        self._last_known_pos = None

    # ──────────────────────────────────────────────────────────────────
    # Debug Accessors
    # ──────────────────────────────────────────────────────────────────

    def get_debug_mask(self):
        """Return debug visualization (None for YOLO — no HSV mask)."""
        return None

    def get_candidates(self):
        """Return list of candidate detections from last detect() call."""
        return []

    def get_last_confidence(self):
        """Return confidence of the last selected detection."""
        return self._last_confidence

    # ──────────────────────────────────────────────────────────────────
    # Backwards Compatibility Stubs
    # ──────────────────────────────────────────────────────────────────

    def set_hand_positions(self, positions):
        """Legacy stub — not used in YOLO path."""
        pass

    def get_motion_mask(self):
        """Legacy stub — not applicable for YOLO."""
        return None

    def get_combined_mask(self):
        """Legacy stub — not applicable for YOLO."""
        return None
