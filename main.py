"""
ShotIQ — Main Application Loop

Complete basketball shot prediction pipeline with YOLOv8 detection:
  YOLOv8 Detection → EKF Tracking → Ball-in-Hand Detection → Trajectory Accumulation
  → EKF Prediction (only when released) → Monte Carlo → SCORE/MISS

Ball-in-hand heuristic (FIXED):
  - Uses average velocity over last 6 frames for smooth detection
  - ball_in_hand = True if smoothed speed < 20 px/frame for 4+ consecutive frames
  - ball_released = True if smoothed speed > 35 px/frame AND was previously held
  - Once released, LOCKED state until ball lost for 15+ frames
  - Minimum 3 real detections before any state decision

Trajectory accumulation:
  - Only adds REAL YOLOv8 detections (not EKF coast predictions)
  - Skips points < 8px apart (stationary/held ball)
  - Only draws arc after 5+ consecutive moving real detections
  - Cleared on EKF reset or user R key

Controls:
  ESC/Q  — Quit
  C      — Calibrate hoop (drag to draw box, ENTER=confirm, ESC=cancel)
  R      — Reset tracker + trajectory
  D      — Toggle YOLO detections debug (not HSV mask)
  S      — Screenshot
  SPACE  — Pause / Resume
"""

import re
import argparse
import time
import cv2
import numpy as np

import config
from detector import BallDetector
from tracker import BallTracker
from trajectory import TrajectoryPredictor
from utils import (
    FPSCounter,
    draw_ball_dot,
    draw_trajectory_trail,
    draw_predicted_arc,
    draw_uncertainty_ellipses,
    draw_telemetry_panel,
    draw_prediction_panel,
    draw_hoop_region,
    draw_fps,
    show_debug_mask,
    draw_no_ball_indicator,
)

# ─────────────────────────────────────────────
# Ball-in-Hand Heuristic Parameters
# ─────────────────────────────────────────────
HELD_SPEED_THRESH = 20.0  # px/frame — slower than this → consider held
RELEASE_SPEED_THRESH = 35.0  # px/frame — faster than this → shot released
HELD_FRAMES_NEEDED = 4  # consecutive slow frames needed to register as held
LOST_FRAMES_TO_UNLOCK = 15  # frames lost before unlocking from "released" state
MIN_REAL_DETS_FOR_STATE = 3  # minimum real detections before state decision

# ─────────────────────────────────────────────
# Trajectory Accumulation Parameters
# ─────────────────────────────────────────────
MIN_MOVE_PX = 8  # minimum pixel movement to add a new trail point
MIN_REAL_DETS = 5  # minimum consecutive real detections before drawing arc
MAX_TRAIL_POINTS = 30  # maximum trajectory points to keep in memory


class BallStateHeuristic:
    """
    Detects ball-in-hand and shot release from pixel velocity only.

    Uses average velocity over last 6 frames for stable detection.
    
    States:
      ball_in_hand=False, released=False  → ball in flight (free)
      ball_in_hand=True                   → ball held/stationary
      ball_in_hand=False, released=True   → shot released (LOCKED until lost)

    Transitions:
      avg speed < 20 px/frame for 4+ frames  → ball_in_hand = True
      speed > 35 px/frame while held         → ball_in_hand = False, released = True
      lost 15+ frames while released         → released = False (unlock)
    """

    def __init__(self):
        """Initialize ball state tracking."""
        self.ball_in_hand = False
        self.released = False  # LOCKED state: once released, stays true until lost 15+ frames
        self._slow_count = 0  # consecutive slow-speed frames
        self._speed_history = []  # last 6 speeds for averaging
        self._real_det_count = 0  # count of real detections
        self._frames_no_detection = 0  # frames since last detection

    def update(self, speed_px_per_frame, has_detection):
        """
        Update ball-in-hand state based on pixel velocity.

        Args:
            speed_px_per_frame: current ball speed in pixels/frame
            has_detection: bool — was the ball actually detected this frame
        """
        # Track real detection count
        if has_detection:
            self._real_det_count += 1
            self._frames_no_detection = 0
        else:
            self._frames_no_detection += 1

        # Don't change state until minimum detections
        if self._real_det_count < MIN_REAL_DETS_FOR_STATE:
            return

        # Unlock "released" state if lost too long
        if self.released and self._frames_no_detection >= LOST_FRAMES_TO_UNLOCK:
            self.released = False
            self.ball_in_hand = False
            self._slow_count = 0
            return

        if not has_detection:
            return

        # Track speed history (last 6 frames)
        self._speed_history.append(speed_px_per_frame)
        if len(self._speed_history) > 6:
            self._speed_history.pop(0)

        # Use average speed over available history
        avg_speed = np.mean(self._speed_history) if self._speed_history else speed_px_per_frame

        # State machine
        if avg_speed < HELD_SPEED_THRESH:
            # Ball moving slowly (held or rolling)
            self._slow_count += 1
            if self._slow_count >= HELD_FRAMES_NEEDED and not self.released:
                # 4+ consecutive slow frames → definitively held (unless already released)
                self.ball_in_hand = True

        elif speed_px_per_frame > RELEASE_SPEED_THRESH and self.ball_in_hand:
            # Ball moving fast and was held → RELEASE EVENT
            self.released = True  # LOCK this state
            self.ball_in_hand = False
            self._slow_count = 0

        else:
            # In the grey zone or not transitioning
            # Gradually decay slow count but don't flip state
            self._slow_count = max(0, self._slow_count - 1)

    def is_held(self):
        """Return True if ball is currently held."""
        return self.ball_in_hand

    def is_released(self):
        """Return True if ball has been released (locked state)."""
        return self.released

    def reset(self):
        """Reset to initial state (used when tracker resets)."""
        self.ball_in_hand = False
        self.released = False
        self._slow_count = 0
        self._speed_history = []
        self._real_det_count = 0
        self._frames_no_detection = 0


class TrailAccumulator:
    """
    Accumulates trajectory trail points from REAL detections only.

    Enforces:
      - Minimum 5px movement between consecutive points
      - 5+ consecutive real detections before drawing
      - Clears when lost detection (coasting/tracking lost)

    This prevents jagged trajectories by:
      1. Excluding EKF coast predictions
      2. Filtering stationary/held ball (< 5px movement)
      3. Requiring a minimum consecutive detection window
    """

    def __init__(self):
        """Initialize trail accumulator."""
        self._points = []  # list of (px, py) real detection positions
        self._consecutive_real = 0  # count of consecutive real detections with movement
        self._last_pt = None  # last added point (for MIN_MOVE_PX check)

    def add(self, px, py, is_real_detection):
        """
        Add a detection to the trajectory trail.

        Args:
            px, py: pixel coordinates
            is_real_detection: bool — True if from detector, False if EKF coast

        Only real detections with sufficient movement are added.
        """
        if not is_real_detection:
            # Coast frame (EKF prediction) — do not add
            # Note: we don't break consecutive count here; that happens in lost_detection()
            return

        # Check minimum movement since last point
        if self._last_pt is not None:
            dist = float(np.hypot(px - self._last_pt[0], py - self._last_pt[1]))
            if dist < MIN_MOVE_PX:
                # Ball held/stationary — skip but don't break count
                return

        # Real detection with sufficient movement
        self._consecutive_real += 1
        self._last_pt = (px, py)

        # Add to trail (keep last N points)
        self._points.append((px, py))
        if len(self._points) > MAX_TRAIL_POINTS:
            self._points.pop(0)

    def get_points(self):
        """
        Get trajectory points for drawing.

        Returns:
            list of (px, py) tuples if consecutive_real >= 5, else empty list
        """
        if self._consecutive_real >= MIN_REAL_DETS:
            return list(self._points)
        return []

    def reset(self):
        """
        Hard reset of trajectory (called by user R key or app reset).

        Clears all points and counters.
        """
        self._points = []
        self._consecutive_real = 0
        self._last_pt = None

    def lost_detection(self):
        """
        Called when no real detection this frame.

        Breaks the consecutive count so trajectory won't be drawn until
        we get 5+ new consecutive real detections.
        """
        self._consecutive_real = 0


# ─────────────────────────────────────────────
# Config Persistence
# ─────────────────────────────────────────────
def save_hoop_to_config(rect):
    try:
        with open("config.py", "r") as f:
            content = f.read()
        x, y, w, h = rect
        content = re.sub(r"^HOOP_X\s*=.*$",      f"HOOP_X = {x}",      content, flags=re.MULTILINE)
        content = re.sub(r"^HOOP_Y\s*=.*$",      f"HOOP_Y = {y}",      content, flags=re.MULTILINE)
        content = re.sub(r"^HOOP_WIDTH\s*=.*$",  f"HOOP_WIDTH = {w}",  content, flags=re.MULTILINE)
        content = re.sub(r"^HOOP_HEIGHT\s*=.*$", f"HOOP_HEIGHT = {h}", content, flags=re.MULTILINE)
        with open("config.py", "w") as f:
            f.write(content)
        config.HOOP_X=x; config.HOOP_Y=y; config.HOOP_WIDTH=w; config.HOOP_HEIGHT=h
        print(f"  Hoop saved.")
        return True
    except Exception as e:
        print(f"  Could not save hoop: {e}")
        return False


# ─────────────────────────────────────────────
# Hoop Calibration (FIXED: Proper State Machine)
# ─────────────────────────────────────────────
class HoopCalibratorApp:
    """
    Interactive hoop calibration with live drag-to-draw feedback.

    State machine: IDLE → DRAWING → DONE
    - LEFT BUTTON DOWN: start drawing
    - MOUSE MOVE (button held): live rectangle feedback
    - LEFT BUTTON UP: finalize rectangle
    """

    def __init__(self):
        self.state = {"drawing": False, "start": None, "end": None, "rect": None}
        self.win = "ShotIQ — Hoop Calibration"

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse event handler (closure over self.state dict)."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state["drawing"] = True
            self.state["start"] = (x, y)
            self.state["end"] = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.state["drawing"]:
            self.state["end"] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.state["drawing"] = False
            if self.state["start"] and self.state["end"]:
                x1 = min(self.state["start"][0], self.state["end"][0])
                y1 = min(self.state["start"][1], self.state["end"][1])
                x2 = max(self.state["start"][0], self.state["end"][0])
                y2 = max(self.state["start"][1], self.state["end"][1])
                self.state["rect"] = (x1, y1, x2 - x1, y2 - y1)

    def calibrate(self, cap):
        """
        Run interactive hoop calibration.

        Returns:
            tuple (x, y, w, h) or None
        """
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._mouse_callback)

        print("  ┌─ HOOP CALIBRATION ─────────────────┐")
        print("  │ Drag to draw hoop box               │")
        print("  │ ENTER = Confirm  ESC = Cancel       │")
        print("  └─────────────────────────────────────┘\n")

        confirmation_time = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            disp = frame.copy()

            # Draw live rectangle while dragging
            if self.state["start"] and self.state["end"]:
                cv2.rectangle(disp, self.state["start"], self.state["end"], 
                            (0, 255, 0), 2, cv2.LINE_AA)

            # Draw confirmation message
            if self.state["rect"]:
                # Show "HOOP SET" for 2 seconds after confirmation
                if confirmation_time is None:
                    confirmation_time = time.time()
                
                elapsed = time.time() - confirmation_time
                if elapsed < 2.0:
                    alpha = 1.0 - (elapsed / 2.0)
                    overlay = disp.copy()
                    cv2.rectangle(overlay, (self.state["rect"][0], self.state["rect"][1]),
                                (self.state["rect"][0] + self.state["rect"][2],
                                 self.state["rect"][1] + self.state["rect"][3]),
                                (0, 255, 0), 3)
                    cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0, disp)

                    # "HOOP SET" text
                    font_scale = 1.5
                    text = "HOOP SET ✓"
                    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                    x = (disp.shape[1] - sz[0]) // 2
                    y = (disp.shape[0] + sz[1]) // 2
                    cv2.putText(disp, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale, (0, 255, 0), 2, cv2.LINE_AA)

            # Instructions
            cv2.putText(disp, "Drag hoop | ENTER=confirm | ESC=cancel",
                       (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(self.win, disp)

            # Key handling
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                cv2.destroyWindow(self.win)
                return None
            elif k == 13 and self.state["rect"]:  # ENTER
                cv2.destroyWindow(self.win)
                return self.state["rect"]

        cv2.destroyWindow(self.win)
        return None


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
class ShotIQApp:

    def __init__(self, args):
        self.args      = args
        self.cap       = None
        self.detector  = BallDetector()
        self.tracker   = BallTracker()
        self.predictor = TrajectoryPredictor()
        self.fps_counter = FPSCounter()
        self.heuristic   = BallStateHeuristic()
        self.trail       = TrailAccumulator()

        self.hoop_rect       = None
        self.paused          = False
        self.debug_mode      = getattr(args, "debug", False)
        self.frame_count     = 0
        self.last_prediction = None
        self.prediction_hold = 0
        self._last_display   = None

        # Frame dimensions (updated on first frame)
        self._frame_w = config.FRAME_WIDTH
        self._frame_h = config.FRAME_HEIGHT

    def initialize_camera(self):
        print("\n  ShotIQ — Basketball Shot Prediction")
        print("  " + "─"*44)
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            print("  ERROR: Cannot open camera.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.FPS_TARGET)
        aw  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_w = aw; self._frame_h = ah
        # Pass frame dims to EKF
        self.tracker._ekf.p2w   # touch to ensure init
        import ekf_tracker as _ekf_mod
        _ekf_mod._FRAME_W = aw
        _ekf_mod._FRAME_H = ah
        print(f"  Camera: {aw}x{ah} @ {fps:.0f} FPS\n")
        return True

    def setup_hoop(self):
        if self.args.calibrate or config.HOOP_X is None:
            rect = HoopCalibratorApp().calibrate(self.cap)
            if rect is None:
                aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.hoop_rect = (aw-200, ah//3, config.HOOP_WIDTH, config.HOOP_HEIGHT)
                print("  Calibration skipped — placeholder hoop. Press C to calibrate.")
            else:
                self.hoop_rect = rect
                ans = input("  Save hoop to config? [Y/n]: ").strip().lower()
                if ans != "n": save_hoop_to_config(rect)
        else:
            self.hoop_rect = (config.HOOP_X, config.HOOP_Y,
                              config.HOOP_WIDTH, config.HOOP_HEIGHT)
            print(f"  Hoop loaded: {self.hoop_rect}")

    def _do_reset(self):
        """Full reset of tracker, trail, heuristic, prediction state."""
        self.tracker.reset()
        self.trail.reset()
        self.heuristic.reset()
        self.detector.clear_last_position()
        self.last_prediction = None
        self.prediction_hold = 0
        print("  Tracker + trail reset.")

    def process_frame(self, frame):
        """
        Main per-frame processing pipeline.

        Steps:
          1. Detect ball using HSV detector
          2. Update EKF tracker with detection (or coast if missing)
          3. Update ball-in-hand state based on pixel velocity
          4. Accumulate trajectory points (only real detections, min 5px apart)
          5. Draw ball position
          6. Draw trajectory arc (only if 5+ real detections and not held)
          7. Run EKF prediction (only if not held)
          8. Draw uncertainty ellipses (only if covariance healthy)
          9. Draw hoop region
         10. Draw status (READY when held, SCORE/MISS when released)
         11. Draw telemetry and FPS

        Returns:
            display frame with all overlays
        """
        self.frame_count += 1
        display = frame.copy()

        # ── 1. Ball detection ──
        detection, det_conf = self.detector.detect(frame)

        # ── 2. EKF tracker update ──
        is_tracking = self.tracker.update(detection, confidence=det_conf)

        # Pixel speed from EKF (px/frame)
        speed_px = self.tracker.get_speed()
        is_real = (detection is not None) and not self.tracker.is_using_prediction()

        # ── 3. Ball-in-hand heuristic ──
        # Update the heuristic with current pixel speed and detection status
        self.heuristic.update(speed_px, has_detection=(detection is not None))
        held = self.heuristic.is_held()
        released = self.heuristic.is_released()

        # ── 4. Trail accumulation ──
        # Only add REAL detections (not EKF predictions) to trajectory
        # Skip points < 8px apart (filtered by TrailAccumulator)
        # Only draw arc after 5+ consecutive real detections
        if is_real:
            # Real detection: add to trajectory trail
            self.trail.add(detection["cx"], detection["cy"], is_real_detection=True)
        else:
            # No real detection (coasting or tracking lost): break consecutive count
            self.trail.lost_detection()

        # ── 5. Draw detected ball ──
        if detection is not None:
            # When held: draw thin circle outline only (not filled blob)
            # When in flight: draw small filled dot at center
            draw_ball_dot(display, detection["cx"], detection["cy"],
                          radius=max(6, detection["radius"]), held=held)

        # ── 6. Draw past trajectory (red arc) — only when in flight ──
        trail_pts = self.trail.get_points()
        if len(trail_pts) >= 2 and not held:
            draw_trajectory_trail(display, trail_pts)

        # ── 7. EKF prediction — only when ball is released ──
        result = {"prediction": None, "confidence": 0.0,
                  "predicted_points": [], "ellipses": [], "Pr": 0.0}

        if released and self.tracker.has_enough_data() and self.hoop_rect is not None:
            raw_pos = self.tracker.get_raw_positions()
            result = self.predictor.predict(raw_pos, self.hoop_rect,
                                            tracker=self.tracker)

            # Draw blue predicted arc
            if result["predicted_points"]:
                draw_predicted_arc(display, result["predicted_points"])

            # Draw red uncertainty ellipses — only if covariance is healthy
            if result.get("ellipses") and self.tracker.covariance_healthy() \
                    and self.tracker.get_lost_frames() == 0:
                draw_uncertainty_ellipses(display, result["ellipses"])

            if result["prediction"] is not None:
                self.last_prediction = result["prediction"]
                self.prediction_hold = 45

        # ── 8. Hoop rectangle ──
        is_score = (result["prediction"] == "SCORE") or \
                   (self.last_prediction == "SCORE" and self.prediction_hold > 0)
        draw_hoop_region(display, self.hoop_rect, is_score=is_score)

        # ── 9. Status panel ──
        if held:
            # Show "READY" when held — no SCORE/MISS
            _draw_ready_label(display, self._frame_w)
        elif result["prediction"] is not None:
            draw_prediction_panel(display, result["prediction"], result["confidence"])
        elif self.prediction_hold > 0:
            draw_prediction_panel(display, self.last_prediction, result["confidence"])
            self.prediction_hold -= 1

        # ── 10. No-ball notice ──
        if not is_tracking and detection is None:
            draw_no_ball_indicator(display, self.tracker.get_frames_missing())

        # ── 11. EKF Telemetry ──
        ekf_state = self.tracker.get_ekf_state()
        draw_telemetry_panel(display, ekf_state, Pr=result.get("Pr", 0.0))

        # ── 12. FPS ──
        self.fps_counter.tick()
        draw_fps(display, self.fps_counter.get_fps())

        # ── 13. Debug mask ──
        if self.debug_mode:
            show_debug_mask(self.detector.get_debug_mask())

        self._last_display = display
        return display

    def run(self):
        if not self.initialize_camera():
            return
        self.setup_hoop()

        win = "ShotIQ — Basketball Shot Prediction"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        print("\n  Controls: ESC/Q=Quit  C=Calibrate  R=Reset  D=Mask  S=Screenshot  SPACE=Pause\n")
        print("  Running… press ESC to quit.\n")

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                display = self.process_frame(frame)
                cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            elif key == ord('c'):
                rect = HoopCalibratorApp().calibrate(self.cap)
                if rect is not None:
                    self.hoop_rect = rect
                    ans = input("  Save hoop to config? [Y/n]: ").strip().lower()
                    if ans != "n": save_hoop_to_config(rect)
            elif key == ord('r'):
                self._do_reset()
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                if not self.debug_mode:
                    cv2.destroyWindow("ShotIQ — HSV Mask")
                print(f"  Debug mask: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('s'):
                if self._last_display is not None:
                    fname = f"shotiq_{int(time.time())}.png"
                    cv2.imwrite(fname, self._last_display)
                    print(f"  Screenshot: {fname}")
            elif key == 32:
                self.paused = not self.paused
                print(f"  {'Paused' if self.paused else 'Resumed'}")

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n  ShotIQ shut down cleanly.\n")


def _draw_ready_label(frame, frame_w):
    """Show 'READY' badge when ball is held."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_w = 200
    px0 = frame_w - panel_w - 10
    py0 = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0), (px0+panel_w, py0+60), (0, 50, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "READY", (px0+16, py0+44),
                font, 1.4, (0, 230, 80), 3, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ShotIQ — Basketball Shot Prediction")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--camera",    type=int, default=None)
    args = parser.parse_args()
    if args.camera is not None:
        config.CAMERA_INDEX = args.camera
    ShotIQApp(args).run()


if __name__ == "__main__":
    main()
