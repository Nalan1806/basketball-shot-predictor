"""
Utility functions for the Basketball Shot Prediction System.

Contains shared helpers for drawing overlays, FPS calculation,
coordinate math, and visual formatting.
"""

import time
import cv2
import numpy as np
import config


# ─────────────────────────────────────────────
# FPS Counter
# ─────────────────────────────────────────────
class FPSCounter:
    """Smooth FPS counter using a rolling window average."""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []
        self.fps = 0.0

    def tick(self):
        """Call once per frame to update the FPS calculation."""
        now = time.time()
        self.timestamps.append(now)
        # Keep only the most recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def get_fps(self):
        return self.fps


# ─────────────────────────────────────────────
# Drawing Utilities
# ─────────────────────────────────────────────
def draw_rounded_rect(frame, pt1, pt2, color, thickness, radius=10):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw the four straight edges (inset by radius)
    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    # Draw the four corner arcs
    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_ball_bbox(frame, x, y, w, h):
    """Draw stylized bounding box around the detected ball."""
    color = config.COLOR_BALL_BBOX
    thickness = config.BBOX_THICKNESS
    corner_len = min(w, h) // 3

    # Draw corner brackets instead of a full rectangle for a modern look
    # Top-left
    cv2.line(frame, (x, y), (x + corner_len, y), color, thickness + 1)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thickness + 1)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness + 1)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness + 1)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness + 1)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness + 1)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness + 1)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness + 1)

    # Draw a subtle center crosshair
    cx, cy = x + w // 2, y + h // 2
    cross_size = 6
    cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), color, 1)
    cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), color, 1)


def draw_trajectory_trail(frame, points):
    """Draw the tracked trajectory points with a fading trail effect."""
    n = len(points)
    if n < 2:
        return

    for i in range(1, n):
        # Fade from transparent to solid — recent points are brighter
        alpha = i / n
        color = tuple(int(c * alpha) for c in config.COLOR_TRAJECTORY)
        radius = max(2, int(config.TRAJECTORY_DOT_RADIUS * alpha))

        cv2.circle(frame, points[i], radius, color, -1)

        # Draw connecting lines between consecutive points
        if i > 0:
            line_color = tuple(int(c * alpha * 0.6) for c in config.COLOR_TRAJECTORY)
            cv2.line(frame, points[i - 1], points[i], line_color, 1, cv2.LINE_AA)


def draw_predicted_trajectory(frame, points):
    """Draw the predicted future trajectory as a dashed curve."""
    n = len(points)
    if n < 2:
        return

    for i in range(1, n):
        # Fade out as we go further into the future
        alpha = max(0.2, 1.0 - (i / n) * 0.8)
        color = tuple(int(c * alpha) for c in config.COLOR_PREDICTION)
        radius = config.PREDICTION_DOT_RADIUS

        # Dashed effect: draw every other segment
        if i % 2 == 0:
            cv2.circle(frame, points[i], radius, color, -1, cv2.LINE_AA)
        cv2.line(frame, points[i - 1], points[i], color, 1, cv2.LINE_AA)


def draw_hoop_region(frame, hoop_rect, is_score=False):
    """Draw the hoop target region with visual feedback."""
    x, y, w, h = hoop_rect

    # Glow effect when scoring
    if is_score:
        # Draw multiple expanding rectangles for glow
        for i in range(3, 0, -1):
            glow_alpha = 0.3 * i
            glow_color = tuple(int(c * glow_alpha) for c in config.COLOR_SCORE)
            cv2.rectangle(frame, (x - i * 3, y - i * 3),
                         (x + w + i * 3, y + h + i * 3), glow_color, 1)

    # Main hoop rectangle
    color = config.COLOR_SCORE if is_score else config.COLOR_HOOP
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.HOOP_THICKNESS)

    # Hoop label
    label = "HOOP"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_x = x + (w - label_size[0]) // 2
    label_y = y - 8
    cv2.putText(frame, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Draw small hoop net lines for visual flair
    net_spacing = w // 6
    for i in range(1, 6):
        nx = x + i * net_spacing
        cv2.line(frame, (nx, y + h), (nx + (i - 3) * 2, y + h + 15),
                color, 1, cv2.LINE_AA)


def draw_prediction_label(frame, prediction, confidence, frame_width):
    """Draw the main SCORE/MISS prediction with confidence bar."""
    if prediction is None:
        return

    is_score = prediction == "SCORE"
    label = prediction
    color = config.COLOR_SCORE if is_score else config.COLOR_MISS

    # --- Background panel ---
    panel_w = 340
    panel_h = 110
    panel_x = frame_width - panel_w - 20
    panel_y = 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  config.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Panel border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), color, 2)

    # Accent line on the left edge
    cv2.line(frame, (panel_x, panel_y), (panel_x, panel_y + panel_h), color, 4)

    # --- Prediction text ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, config.FONT_SCALE_LARGE, 3)[0]
    text_x = panel_x + 20
    text_y = panel_y + 50

    # Text shadow
    cv2.putText(frame, label, (text_x + 2, text_y + 2),
                font, config.FONT_SCALE_LARGE, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (text_x, text_y),
                font, config.FONT_SCALE_LARGE, color, 3, cv2.LINE_AA)

    # --- Confidence bar ---
    bar_x = panel_x + 20
    bar_y = panel_y + 70
    bar_w = panel_w - 40
    bar_h = 12

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)

    # Fill
    fill_w = int(bar_w * confidence)
    bar_color = config.COLOR_CONFIDENCE_HIGH if confidence > 0.6 else config.COLOR_CONFIDENCE_LOW
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  bar_color, -1)

    # Confidence text
    conf_text = f"Confidence: {confidence:.0%}"
    cv2.putText(frame, conf_text, (bar_x, bar_y + bar_h + 18),
                font, config.FONT_SCALE_SMALL, config.COLOR_INFO, 1, cv2.LINE_AA)


def draw_fps(frame, fps):
    """Draw FPS counter in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background pill
    text = f"FPS: {fps:.0f}"
    text_size = cv2.getTextSize(text, font, config.FONT_SCALE_MEDIUM, 2)[0]
    pad = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10),
                  (10 + text_size[0] + 2 * pad, 10 + text_size[1] + 2 * pad),
                  config.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text, (10 + pad, 10 + text_size[1] + pad),
                font, config.FONT_SCALE_MEDIUM, config.COLOR_FPS, 2, cv2.LINE_AA)


def draw_status_bar(frame, status_text, frame_height, frame_width):
    """Draw a bottom status bar with system info."""
    bar_h = 35
    bar_y = frame_height - bar_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (frame_width, frame_height),
                  config.COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Status text
    cv2.putText(frame, status_text, (15, bar_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_SMALL,
                config.COLOR_INFO, 1, cv2.LINE_AA)

    # Brand text on the right
    brand = "ShotIQ v1.0"
    brand_size = cv2.getTextSize(brand, cv2.FONT_HERSHEY_SIMPLEX,
                                  config.FONT_SCALE_SMALL, 1)[0]
    cv2.putText(frame, brand, (frame_width - brand_size[0] - 15, bar_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_SMALL,
                (100, 200, 255), 1, cv2.LINE_AA)


def draw_no_ball_indicator(frame, frames_missing):
    """Draw a subtle indicator when the ball is not detected."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Searching for ball... ({frames_missing} frames)"
    text_size = cv2.getTextSize(text, font, config.FONT_SCALE_SMALL, 1)[0]

    x = (frame.shape[1] - text_size[0]) // 2
    y = 60

    # Pulsing dot animation based on frame count
    dot_color = (0, 150, 255) if (frames_missing // 4) % 2 == 0 else (0, 80, 150)
    cv2.circle(frame, (x - 15, y - 5), 5, dot_color, -1)

    cv2.putText(frame, text, (x, y), font, config.FONT_SCALE_SMALL,
                (150, 150, 150), 1, cv2.LINE_AA)


def smooth_positions(positions, window_size=None):
    """Apply moving average smoothing to a list of (x, y) positions."""
    if window_size is None:
        window_size = config.SMOOTHING_WINDOW

    if len(positions) < window_size:
        return positions

    smoothed = []
    for i in range(len(positions)):
        start = max(0, i - window_size + 1)
        window = positions[start:i + 1]
        avg_x = int(np.mean([p[0] for p in window]))
        avg_y = int(np.mean([p[1] for p in window]))
        smoothed.append((avg_x, avg_y))

    return smoothed


def distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
