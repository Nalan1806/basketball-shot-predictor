"""
Drawing Utilities — ShotIQ

Clean visual overlays for the basketball shot prediction system.
"""

import cv2
import numpy as np


class FPSCounter:
    """Exponential moving average FPS counter."""
    def __init__(self, alpha=0.1):
        import time
        self._alpha = alpha
        self._fps = 0.0
        self._last = time.time()

    def tick(self):
        import time
        now = time.time()
        dt = now - self._last
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


def draw_ball_dot(frame, cx, cy, radius=10, held=False):
    """Draw the detected ball indicator."""
    cx, cy = int(cx), int(cy)
    if held:
        cv2.circle(frame, (cx, cy), radius, (0, 220, 0), 2, cv2.LINE_AA)
    else:
        cv2.circle(frame, (cx, cy), 6, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, (0, 180, 0), 1, cv2.LINE_AA)


def draw_trajectory_trail(frame, positions, max_points=30):
    """Draw past positions as a smooth fading red arc with glow."""
    pts = list(positions)[-max_points:]
    if len(pts) < 2:
        return
    pts_smooth = smooth_positions(pts, window=7)  # Increased smoothing window for cinematic fluid movement
    pts_int = np.int32(pts_smooth)
    n = len(pts_int)
    
    # Draw glow underneath
    for i in range(1, n):
        alpha = int(100 + 100 * (i / n))
        cv2.line(frame, tuple(pts_int[i-1]), tuple(pts_int[i]),
                 (0, 0, alpha), 8 if i > n // 2 else 4, cv2.LINE_AA)
                 
    # Draw core line
    for i in range(1, n):
        alpha = int(150 + 105 * (i / n))
        cv2.line(frame, tuple(pts_int[i-1]), tuple(pts_int[i]),
                 (0, 50, alpha), 2, cv2.LINE_AA)
                 
    cv2.circle(frame, tuple(pts_int[-1]), 5, (0, 100, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(pts_int[-1]), 9, (0, 0, 180), 2, cv2.LINE_AA)


def draw_predicted_arc(frame, predicted_points, max_points=80):
    """Draw the forward projection as a fading glowing blue polyline."""
    pts = predicted_points[:max_points]
    if len(pts) < 2:
        return
    pts_int = np.int32(pts)
    n = len(pts_int)
    
    # Draw glow
    for i in range(1, n):
        alpha = int(150 * (1.0 - 0.7 * i / n))
        cv2.line(frame, tuple(pts_int[i-1]), tuple(pts_int[i]),
                 (alpha, int(alpha * 0.3), 0), 6, cv2.LINE_AA)
                 
    # Draw core
    for i in range(1, n):
        alpha = int(255 * (1.0 - 0.7 * i / n))
        thickness = 2
        cv2.line(frame, tuple(pts_int[i-1]), tuple(pts_int[i]),
                 (alpha, int(alpha * 0.6), 50), thickness, cv2.LINE_AA)
                 
    for i in range(0, n, 6):
        cv2.circle(frame, tuple(pts_int[i]), 3, (255, 150, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(pts_int[i]), 6, (150, 50, 0), 1, cv2.LINE_AA)
        
    cv2.circle(frame, tuple(pts_int[-1]), 7, (255, 200, 50), -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(pts_int[-1]), 12, (255, 100, 0), 2, cv2.LINE_AA)


def draw_telemetry_panel(frame, telemetry_data):
    """Sleek cinematic top-left panel showing system telemetry."""
    if not telemetry_data:
        return

    lines = []
    lines.append(("SHOT IQ | ANALYTICS", (255, 255, 255), 0.6, 2))
    
    # Process fields
    for k, v in telemetry_data.items():
        if k == "FPS":
            lines.append((f"PROCESSING: {v:.0f} FPS", (150, 150, 150), 0.45, 1))
        elif k == "Position":
            lines.append((f"TARGET POS: ({v[0]:.0f}, {v[1]:.0f}) px", (200, 200, 255), 0.45, 1))
        elif k == "Velocity":
            lines.append((f"SPEED: {np.hypot(v[0], v[1]):.0f} px/s", (200, 200, 255), 0.45, 1))
        elif k == "Confidence":
            lines.append((f"AI CONFIDENCE: {v:.0%}", (200, 200, 200), 0.45, 1))
        elif k == "State":
            color = (0, 220, 80) # Default green
            if v in ["SEARCHING", "BALL DETECTED"]: color = (0, 150, 255) # Orange
            elif v == "AIMING": color = (0, 220, 255) # Yellow
            elif v in ["SHOT RELEASED", "ANALYZING TRAJECTORY", "predicting..."]: color = (255, 150, 0) # Blueish
            elif v in ["ON TARGET", "SHOT!", "will go in"]: color = (0, 255, 100) # Bright green
            elif v in ["OFF TARGET", "MISS"]: color = (0, 50, 255) # Red
            lines.append((f"STATUS: {v.upper()}", color, 0.55, 2))

    font = cv2.FONT_HERSHEY_DUPLEX
    pad = 20
    line_h = 28
    panel_w = 320
    panel_h = len(lines) * line_h + 2 * pad
    
    px0, py0 = 20, 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0), (px0 + panel_w, py0 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Sleek left accent border
    cv2.line(frame, (px0, py0), (px0, py0 + panel_h), (0, 120, 255), 4)

    for i, (text, color, scale, thick) in enumerate(lines):
        ty = py0 + pad + (i + 1) * line_h - 10
        cv2.putText(frame, text, (px0 + pad, ty), font, scale, color, thick, cv2.LINE_AA)


def draw_prediction_panel(frame, prediction, confidence):
    """Cinematic top-right panel for prediction results."""
    if prediction is None:
        return
    h_frame, w_frame = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    
    is_score = (prediction in ["ON TARGET", "SHOT!", "will go in"])
    label_color = (0, 255, 100) if is_score else (0, 50, 255)
    bg_color = (0, 30, 10) if is_score else (20, 0, 0)
    bar_color = (0, 200, 80) if is_score else (0, 50, 220)
    border_color = (0, 200, 50) if is_score else (0, 30, 200)
    
    panel_w, panel_h = 340, 110
    px0 = w_frame - panel_w - 20
    py0 = 20
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0), (px0 + panel_w, py0 + panel_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Border
    cv2.rectangle(frame, (px0, py0), (px0 + panel_w, py0 + panel_h), border_color, 2, cv2.LINE_AA)
    
    # Text
    cv2.putText(frame, "AI PREDICTION:", (px0 + 20, py0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, prediction, (px0 + 20, py0 + 70), font, 1.4, label_color, 3, cv2.LINE_AA)
    
    # Confidence Bar
    cv2.putText(frame, f"CONFIDENCE {confidence:.0%}", (px0 + 20, py0 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    bar_x, bar_y = px0 + 150, py0 + 90
    bar_max_w, bar_h = panel_w - 170, 6
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_max_w * confidence)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)


def draw_hoop_region(frame, hoop_rect, is_score=False):
    """Draw the calibrated hoop rectangle."""
    if hoop_rect is None:
        return
    x, y, w, h = hoop_rect
    color = (0, 255, 80) if is_score else (0, 200, 0)
    thickness = 3 if is_score else 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
    cx, cy = x + w // 2, y + h // 2
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
    cv2.putText(frame, "HOOP", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


def show_debug_mask(mask, window_name="ShotIQ — Debug"):
    """Show a debug mask in a separate window."""
    if mask is None:
        return
    display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(display, "Debug (D to hide)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(window_name, display)
