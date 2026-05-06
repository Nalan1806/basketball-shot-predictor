"""
Drawing Utilities — ShotIQ

Draws exactly 8 visual elements on the frame. Nothing else.

Elements:
  1. draw_ball_dot         — small green circle at detected ball
  2. draw_trajectory_trail — red arc of past positions
  3. draw_predicted_arc    — blue arc of EKF future positions
  4. draw_uncertainty_ellipses — red EKF confidence ellipses (axes clamped to 80px)
  5. draw_telemetry_panel  — bottom-left dark panel: pos, vel, speed
  6. draw_prediction_panel — top-right: SCORE/MISS + confidence bar
  7. draw_hoop_region      — green rectangle when calibrated
  8. draw_fps              — top-left FPS counter

Also includes:
  - FPSCounter utility class
  - smooth_positions utility
  - distance utility
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
class FPSCounter:
    """Exponential moving average FPS counter."""

    def __init__(self, alpha=0.1):
        import time
        self._alpha = alpha
        self._fps   = 0.0
        self._last  = time.time()

    def tick(self):
        import time
        now = time.time()
        dt  = now - self._last
        self._last = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = self._alpha * instant + (1 - self._alpha) * self._fps

    def get_fps(self):
        return self._fps


def smooth_positions(positions, window=5):
    """Simple moving-average smoother over a list of (x, y) tuples."""
    if len(positions) < 2:
        return positions

    smoothed = []
    for i in range(len(positions)):
        lo = max(0, i - window // 2)
        hi = min(len(positions), i + window // 2 + 1)
        chunk = positions[lo:hi]
        avg_x = int(sum(p[0] for p in chunk) / len(chunk))
        avg_y = int(sum(p[1] for p in chunk) / len(chunk))
        smoothed.append((avg_x, avg_y))
    return smoothed


def distance(p1, p2):
    """Euclidean distance between two 2-D points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


# ─────────────────────────────────────────────
# 1. Ball dot
# ─────────────────────────────────────────────
def draw_ball_dot(frame, cx, cy, radius=10, held=True):
    """
    Draw the detected ball:
    - When held (held=True): thin circle OUTLINE only (thickness=2), no filled blob
    - When in flight (held=False): small filled dot at center (radius=4)
    """
    cx, cy = int(cx), int(cy)
    if held:
        # When held: draw thin outline (not filled blob)
        cv2.circle(frame, (cx, cy), radius, (0, 220, 0), 2, cv2.LINE_AA)
    else:
        # When in flight: draw small filled dot at center
        cv2.circle(frame, (cx, cy), 4, (0, 220, 0), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# 2. Past trajectory (red arc)
# ─────────────────────────────────────────────
def draw_trajectory_trail(frame, positions, max_points=30):
    """
    Draw the last N detected ball positions as a smooth red arc.
    Uses moving-average smoothing (window=3) and cv2.polylines for clean rendering.
    """
    pts = list(positions)[-max_points:]
    if len(pts) < 2:
        return

    # Apply moving-average smoothing to reduce jitter
    pts_smooth = smooth_positions(pts, window=3)
    pts_smooth = np.int32(pts_smooth)
    
    # Draw as smooth polyline with anti-aliasing
    cv2.polylines(frame, [pts_smooth], False, (0, 0, 255), 2, cv2.LINE_AA)

    # Bright red dot at the most recent position
    cv2.circle(frame, pts[-1], 4, (0, 0, 255), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# 3. Predicted future arc (blue)
# ─────────────────────────────────────────────
def draw_predicted_arc(frame, predicted_points, max_points=80):
    """
    Draw the EKF forward projection as a smooth blue polyline.
    Add small filled circles every 5th point for clarity.
    """
    pts = predicted_points[:max_points]
    if len(pts) < 2:
        return

    # Convert to int32 for polylines
    pts_int = np.int32(pts)
    
    # Draw smooth blue polyline
    cv2.polylines(frame, [pts_int], False, (255, 100, 0), 2, cv2.LINE_AA)

    # Draw small filled circles every 5th point for visual markers
    for i in range(0, len(pts), 5):
        cv2.circle(frame, pts_int[i], 3, (255, 100, 0), -1, cv2.LINE_AA)

    # Mark the end of the predicted arc
    cv2.circle(frame, pts_int[-1], 5, (255, 180, 0), -1, cv2.LINE_AA)


# Keep old name for backwards compat with trajectory.py
def draw_predicted_trajectory(frame, predicted_points):
    draw_predicted_arc(frame, predicted_points)


# ─────────────────────────────────────────────
# 4. Uncertainty ellipses (red, clamped to 80px)
# ─────────────────────────────────────────────
MAX_ELLIPSE_AXIS = 80   # pixels — hard cap so they stay readable

def draw_uncertainty_ellipses(frame, ellipses, color=(0, 0, 200)):
    """
    Draw 66% confidence ellipses along the predicted arc.
    Axes are clamped to MAX_ELLIPSE_AXIS to prevent giant blobs.
    """
    h, w = frame.shape[:2]
    for ell in ellipses:
        if ell is None:
            continue
        cx, cy = int(ell["center"][0]), int(ell["center"][1])
        # Clamp axes
        a = int(min(ell["axes"][0], MAX_ELLIPSE_AXIS))
        b = int(min(ell["axes"][1], MAX_ELLIPSE_AXIS))
        angle = float(ell["angle"])

        if a < 2 or b < 2:
            continue
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        try:
            cv2.ellipse(frame, (cx, cy), (a, b), angle,
                        0, 360, color, 1, cv2.LINE_AA)
        except cv2.error:
            pass


# ─────────────────────────────────────────────
# 5. EKF Telemetry panel (bottom-left)
# ─────────────────────────────────────────────
def draw_telemetry_panel(frame, ekf_state, Pr=0.0):
    """
    Bottom-left semi-transparent panel showing EKF state.

    Args:
        ekf_state: dict from EKFBallTracker.get_ekf_state()
        Pr:        Monte Carlo shot probability float 0-1
    """
    if ekf_state is None:
        return

    px_m   = ekf_state.get("px_m",    0.0)
    py_m   = ekf_state.get("py_m",    0.0)
    vx_ms  = ekf_state.get("vx_ms",   0.0)
    vy_ms  = ekf_state.get("vy_ms",   0.0)
    speed  = ekf_state.get("speed_ms", 0.0)

    lines = [
        ("EKF Telemetry",              (255, 255, 255), 0.52, 2),
        (f"Pos:   ({px_m:+.2f}, {py_m:+.2f}) m",  (160, 160, 255), 0.46, 1),
        (f"Vel:   ({vx_ms:+.2f}, {vy_ms:+.2f}) m/s", (160, 160, 255), 0.46, 1),
        (f"Speed: {speed:.2f} m/s",    (160, 160, 255), 0.46, 1),
        (f"Pr = {Pr:.3f}",             (60, 80, 255),   0.50, 2),
    ]

    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 10
    line_h = 22
    panel_w = 265
    panel_h = len(lines) * line_h + 2 * pad

    h_frame = frame.shape[0]
    px0 = 10
    py0 = h_frame - panel_h - 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0),
                  (px0 + panel_w, py0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Red left-edge accent
    cv2.line(frame, (px0, py0), (px0, py0 + panel_h), (0, 0, 200), 3)

    for i, (text, color, scale, thick) in enumerate(lines):
        y = py0 + pad + (i + 1) * line_h - 4
        cv2.putText(frame, text, (px0 + pad, y),
                    font, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
# 6. Prediction panel (top-right)
# ─────────────────────────────────────────────
def draw_prediction_panel(frame, prediction, confidence):
    """
    Top-right panel: large SCORE/MISS label + confidence bar.

    Args:
        prediction:  "SCORE" | "MISS" | None
        confidence:  float 0-1
    """
    if prediction is None:
        return

    h_frame, w_frame = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_score = (prediction == "SCORE")
    label_color  = (0, 220, 60)   if is_score else (0, 50, 240)
    bg_color     = (0, 60, 10)    if is_score else (20, 10, 60)
    bar_color    = (0, 200, 60)   if is_score else (0, 50, 220)

    panel_w = 230
    panel_h = 85
    px0 = w_frame - panel_w - 10
    py0 = 10

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0),
                  (px0 + panel_w, py0 + panel_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Label
    cv2.putText(frame, prediction,
                (px0 + 14, py0 + 48),
                font, 1.5, label_color, 3, cv2.LINE_AA)

    # Confidence text
    conf_text = f"Confidence: {confidence:.0%}"
    cv2.putText(frame, conf_text,
                (px0 + 14, py0 + 68),
                font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    # Confidence bar
    bar_x = px0 + 14
    bar_y = py0 + 73
    bar_max_w = panel_w - 28
    bar_h = 6
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_max_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_max_w * confidence)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)


# Keep old name used by trajectory.py
def draw_prediction_label(frame, prediction, confidence, frame_w):
    draw_prediction_panel(frame, prediction, confidence)


# ─────────────────────────────────────────────
# 7. Hoop rectangle (green)
# ─────────────────────────────────────────────
def draw_hoop_region(frame, hoop_rect, is_score=False):
    """
    Draw the calibrated hoop rectangle.
    Flashes brighter green on a SCORE prediction.
    """
    if hoop_rect is None:
        return
    x, y, w, h = hoop_rect
    color = (0, 255, 80) if is_score else (0, 200, 0)
    thickness = 3 if is_score else 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

    # Cross-hair at centre
    cx, cy = x + w // 2, y + h // 2
    cv2.drawMarker(frame, (cx, cy), color,
                   cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

    # "HOOP" label
    cv2.putText(frame, "HOOP", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# 8. FPS counter (top-left)
# ─────────────────────────────────────────────
def draw_fps(frame, fps):
    """Draw FPS in the top-left corner."""
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (180, 180, 180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Debug mask window
# ─────────────────────────────────────────────
def show_debug_mask(mask, window_name="ShotIQ — HSV Mask"):
    """Show the HSV detection mask in a separate window."""
    if mask is None:
        return
    display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(display, "HSV Mask (D to hide)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(window_name, display)


# ─────────────────────────────────────────────
# Stubs kept for backwards compat
# ─────────────────────────────────────────────
def draw_ball_bbox(frame, x, y, w, h):
    """Legacy — replaced by draw_ball_dot; kept so imports don't break."""
    pass

def draw_status_bar(frame, text, frame_h, frame_w):
    """Legacy stub — status bar removed from clean UI."""
    pass

def draw_no_ball_indicator(frame, frames_missing):
    """Show a small 'No Ball' notice when tracking is lost."""
    if frames_missing < 5:
        return
    cv2.putText(frame, f"No ball ({frames_missing}f)",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (100, 100, 255), 1, cv2.LINE_AA)
