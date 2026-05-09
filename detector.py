"""
YOLOv8 Basketball Detector — ShotIQ

Stable real-time detection using YOLOv8n (nano) model:
  - Class 32 (sports ball) detection only
  - Confidence threshold from config
  - Tight proximity gate (80px) to prevent false snapping
  - Single-frame outlier rejection: requires 2 consecutive detections
    at a new location before accepting a tracking jump
  - Falls back to closest-to-last-position for smooth tracking
"""

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

import config


class BallDetector:
    """
    YOLOv8-based basketball detector with outlier rejection.

    Features:
      - YOLOv8n nano model for fast CPU inference
      - COCO class 32 (sports ball) filter
      - Proximity gate: MAX_JUMP_PX from last known position
      - Outlier hysteresis: requires OUTLIER_FRAMES_REQUIRED consecutive
        detections at a distant location before accepting as new target
    """

    def __init__(self):
        """Initialize YOLO detector and load model."""
        self._last_known_pos = None     # (px, py) for proximity gate
        self._outlier_candidate = None  # (px, py) candidate during hysteresis
        self._outlier_count = 0         # consecutive frames at outlier position

        # Load YOLOv8n model
        try:
            self.model = YOLO(config.YOLO_MODEL_PATH)
            print(f"  YOLOv8 model '{config.YOLO_MODEL_PATH}' loaded successfully")
        except Exception as e:
            print(f"  ERROR loading YOLOv8 model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Detect basketball using YOLOv8.

        Process:
          1. Run YOLOv8 inference
          2. Filter to COCO class 32 (sports ball)
          3. Apply proximity gate (MAX_JUMP_PX)
          4. Apply outlier hysteresis (reject single-frame jumps)
          5. Return best detection

        Args:
            frame: BGR image from OpenCV

        Returns:
            tuple: (detection_dict or None, confidence: float)
              detection_dict keys: cx, cy, x, y, w, h, radius
        """
        if self.model is None:
            return None, 0.0

        try:
            results = self.model(frame, conf=config.YOLO_CONFIDENCE, verbose=False)
            boxes = results[0].boxes

            # Filter to sports ball class
            candidates = []
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                if class_id != config.YOLO_SPORTS_BALL_CLASS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                radius = max(w, h) // 2

                candidates.append({
                    "cx": cx, "cy": cy,
                    "x": x1, "y": y1,
                    "w": w, "h": h,
                    "radius": radius,
                    "conf": conf,
                })

            if not candidates:
                return None, 0.0

            # ── Proximity gate + outlier rejection ──
            if self._last_known_pos is not None:
                lx, ly = self._last_known_pos

                # Split into nearby vs distant
                nearby = []
                distant = []
                for c in candidates:
                    dist = np.hypot(c["cx"] - lx, c["cy"] - ly)
                    if dist <= config.MAX_JUMP_PX:
                        nearby.append(c)
                    else:
                        distant.append(c)

                if nearby:
                    # Good: pick closest to last position
                    best = min(nearby,
                               key=lambda c: np.hypot(c["cx"] - lx, c["cy"] - ly))
                    self._outlier_candidate = None
                    self._outlier_count = 0

                elif distant:
                    # All detections are far — apply hysteresis
                    best_distant = max(distant, key=lambda c: c["conf"])
                    bd_pos = (best_distant["cx"], best_distant["cy"])

                    if self._outlier_candidate is not None:
                        od = np.hypot(bd_pos[0] - self._outlier_candidate[0],
                                      bd_pos[1] - self._outlier_candidate[1])
                        if od < config.MAX_JUMP_PX:
                            self._outlier_count += 1
                        else:
                            # Different distant position — restart count
                            self._outlier_candidate = bd_pos
                            self._outlier_count = 1
                    else:
                        self._outlier_candidate = bd_pos
                        self._outlier_count = 1

                    if self._outlier_count >= config.OUTLIER_FRAMES_REQUIRED:
                        # Accept the new position (ball genuinely moved far)
                        best = best_distant
                        self._outlier_candidate = None
                        self._outlier_count = 0
                    else:
                        # Reject — single-frame outlier
                        return None, 0.0
                else:
                    return None, 0.0
            else:
                # No tracking history — pick highest confidence
                best = max(candidates, key=lambda c: c["conf"])

            confidence = float(best["conf"])
            self._last_known_pos = (best["cx"], best["cy"])

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
            return None, 0.0

    def clear_last_position(self):
        """Reset proximity gate (called after tracker reset)."""
        self._last_known_pos = None
        self._outlier_candidate = None
        self._outlier_count = 0

    def get_debug_mask(self):
        """Return debug visualization (None for YOLO — no HSV mask)."""
        return None
