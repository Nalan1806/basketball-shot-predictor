"""
ShotIQ — AI Basketball Shot Analysis System
Offline Video Processing Mode

Cinematic offline analysis of basketball shots.
  YOLOv8 Detection → Kalman Tracking → Gravity Prediction → SCORE/MISS
"""

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
    draw_telemetry_panel,
    draw_prediction_panel,
    draw_hoop_region,
)

# ─────────────────────────────────────────────
# Ball-in-Hand Heuristic Parameters
# ─────────────────────────────────────────────
HELD_SPEED_THRESH = 200.0     # Scaled dynamically based on video height
RELEASE_SPEED_THRESH = 500.0  
HELD_FRAMES_NEEDED = 4        
LOST_FRAMES_TO_UNLOCK = 15    
MIN_REAL_DETS_FOR_STATE = 3   

# ─────────────────────────────────────────────
# Trajectory Accumulation Parameters
# ─────────────────────────────────────────────
MIN_MOVE_PX = 8
MIN_REAL_DETS = 5
MAX_TRAIL_POINTS = 30


class BallStateHeuristic:
    """Detects ball-in-hand and shot release from pixel velocity."""

    def __init__(self):
        self.ball_in_hand = False
        self.released = False
        self._slow_count = 0
        self._speed_history = []
        self._real_det_count = 0
        self._frames_no_detection = 0

    def update(self, speed_px_per_sec, has_detection):
        if has_detection:
            self._real_det_count += 1
            self._frames_no_detection = 0
        else:
            self._frames_no_detection += 1

        if self._real_det_count < MIN_REAL_DETS_FOR_STATE:
            return

        if self.released and self._frames_no_detection >= LOST_FRAMES_TO_UNLOCK:
            self.released = False
            self.ball_in_hand = False
            self._slow_count = 0
            return

        if not has_detection:
            return

        self._speed_history.append(speed_px_per_sec)
        if len(self._speed_history) > 6:
            self._speed_history.pop(0)

        avg_speed = np.mean(self._speed_history) if self._speed_history else speed_px_per_sec

        if avg_speed < HELD_SPEED_THRESH:
            self._slow_count += 1
            if self._slow_count >= HELD_FRAMES_NEEDED and not self.released:
                self.ball_in_hand = True
        elif speed_px_per_sec > RELEASE_SPEED_THRESH and self.ball_in_hand:
            self.released = True
            self.ball_in_hand = False
            self._slow_count = 0
        else:
            self._slow_count = max(0, self._slow_count - 1)

    def is_held(self):
        return self.ball_in_hand

    def is_released(self):
        return self.released

    def reset(self):
        self.ball_in_hand = False
        self.released = False
        self._slow_count = 0
        self._speed_history = []
        self._real_det_count = 0
        self._frames_no_detection = 0


class TrailAccumulator:
    """Accumulates trajectory trail points from real detections only."""

    def __init__(self):
        self._points = []
        self._consecutive_real = 0
        self._last_pt = None

    def add(self, px, py, is_real_detection):
        if not is_real_detection:
            return
        if self._last_pt is not None:
            dist = float(np.hypot(px - self._last_pt[0], py - self._last_pt[1]))
            if dist < MIN_MOVE_PX:
                return
        self._consecutive_real += 1
        self._last_pt = (px, py)
        self._points.append((px, py))
        if len(self._points) > MAX_TRAIL_POINTS:
            self._points.pop(0)

    def get_points(self):
        if self._consecutive_real >= MIN_REAL_DETS:
            return list(self._points)
        return []

    def reset(self):
        self._points = []
        self._consecutive_real = 0
        self._last_pt = None

    def lost_detection(self):
        self._consecutive_real = 0


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
class ShotIQApp:

    def __init__(self, args):
        self.args = args
        self.cap = None
        self.detector = BallDetector()
        self.tracker = BallTracker()
        self.predictor = TrajectoryPredictor()
        self.fps_counter = FPSCounter()
        self.heuristic = BallStateHeuristic()
        self.trail = TrailAccumulator()

        self.hoop_rect = None
        self.frame_count = 0
        self.last_prediction = None
        self.prediction_hold = 0
        self.post_shot_timer = 0
        self.final_shot_result = None
        self.persistent_state = "SEARCHING"
        self._last_display = None
        self._last_detection = None
        self._last_det_conf = 0.0

        self._frame_w = 1280
        self._frame_h = 720
        self.video_fps = 30.0

    def initialize_video(self):
        print("\n  ShotIQ — Offline Video Analysis")
        print("  " + "-" * 44)
        
        input_path = getattr(self.args, "input", None)
        if not input_path:
            print("  ERROR: Please provide an input video using --input")
            return False
            
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            print(f"  ERROR: Cannot open video: {input_path}")
            return False
            
        self._frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30.0
            
        print(f"  Input Video: {self._frame_w}x{self._frame_h} @ {self.video_fps:.2f} FPS")
        
        # Scale physics and hoop parameters to match video resolution (relative to 720p base)
        scale_factor = self._frame_h / 720.0
        config.GRAVITY_PX_PER_S2 = 500.0 * scale_factor
        config.HOOP_WIDTH = int(120 * scale_factor)
        config.HOOP_HEIGHT = int(80 * scale_factor)
        config.INTERSECTION_TOLERANCE = int(60 * scale_factor)
        config.MAX_JUMP_PX = int(80 * scale_factor)
        
        global HELD_SPEED_THRESH, RELEASE_SPEED_THRESH
        HELD_SPEED_THRESH = 150.0 * scale_factor
        RELEASE_SPEED_THRESH = 300.0 * scale_factor
        
        print(f"  Physics auto-scaled for {self._frame_h}p (Gravity: {config.GRAVITY_PX_PER_S2:.1f} px/s^2)")
        return True

    def setup_hoop(self):
        """Manually set hoop region to match the left backboard in the video."""
        print("  Manually setting hoop region to the backboard area...")
        # Rim is around (80, 362). We center the hoop_rect around it.
        # This explicitly aligns with the backboard on the left side of the frame.
        self.hoop_rect = (
            40, 
            230, 
            config.HOOP_WIDTH, 
            config.HOOP_HEIGHT
        )
        print(f"  Hoop region manually set to: {self.hoop_rect}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def process_frame(self, frame, dt):
        """Main per-frame processing pipeline."""
        self.frame_count += 1
        display = frame.copy()
        self.fps_counter.tick()

        # ── 1. Ball detection (Skip frames for FPS optimization) ──
        if self.frame_count % getattr(config, "YOLO_SKIP_FRAMES", 1) == 0:
            detection, det_conf = self.detector.detect(frame)
            self._last_detection = detection
            self._last_det_conf = det_conf
            has_detection = (detection is not None)
        else:
            detection = None
            det_conf = self._last_det_conf
            has_detection = (self._last_detection is not None)

        # ── 2. Kalman tracker update ──
        # Tracker uses 'None' detection to coast/interpolate smoothly
        is_tracking = self.tracker.update(detection, confidence=det_conf, dt=dt)
        tracker_state = self.tracker.get_state()

        # ── 3. Ball-in-hand heuristic (using px/s speed) ──
        speed_px_s = tracker_state["speed"]
        self.heuristic.update(speed_px_s, has_detection=has_detection)
        held = self.heuristic.is_held()
        released = self.heuristic.is_released()

        # ── 4. Trail accumulation (Use smoothed Kalman position for trail) ──
        if is_tracking and not self.tracker.is_using_prediction():
            self.trail.add(tracker_state["x"], tracker_state["y"], is_real_detection=True)
        elif is_tracking and self.frame_count % getattr(config, "YOLO_SKIP_FRAMES", 1) != 0:
            self.trail.add(tracker_state["x"], tracker_state["y"], is_real_detection=True)
        else:
            self.trail.lost_detection()

        # ── 5. Draw detected ball (or interpolated ball) ──
        ball_pos = (tracker_state["x"], tracker_state["y"]) if is_tracking else None
        if ball_pos is not None:
            radius = max(6, detection["radius"]) if detection else 6
            draw_ball_dot(display, ball_pos[0], ball_pos[1], radius=radius, held=held)

        # ── 6. Draw past trajectory (red arc) ──
        trail_pts = self.trail.get_points()
        if len(trail_pts) >= 2 and not held:
            draw_trajectory_trail(display, trail_pts)

        # ── 7. Trajectory prediction (only when released) ──
        result = {"prediction": None, "confidence": 0.0, "predicted_points": []}
        
        vy = tracker_state.get("vy", 0.0)
        is_moving_up = vy < -10  # negative Y is UP in pixel coordinates
        
        can_predict = released and self.tracker.has_enough_data() and self.hoop_rect is not None

        if can_predict:
            positions = self.tracker.get_positions()
            result = self.predictor.predict(positions, self.hoop_rect, dt=dt)

            if result["predicted_points"]:
                draw_predicted_arc(display, result["predicted_points"])

            if result["prediction"] is not None:
                self.last_prediction = result["prediction"]
                self.prediction_hold = 45

        # Determine logical system state
        if self.post_shot_timer > 0:
            system_state = self.final_shot_result
            self.post_shot_timer -= 1
        else:
            system_state = "SEARCHING"
            if is_tracking:
                if held:
                    system_state = "AIMING"
                elif released:
                    if result["prediction"] == "ON TARGET":
                        system_state = "will go in"
                        self.persistent_state = system_state
                    elif result["prediction"] is not None:
                        system_state = result["prediction"]
                        self.persistent_state = system_state
                    elif self.prediction_hold > 0:
                        system_state = self.persistent_state
                    elif result["predicted_points"]:
                        system_state = "predicting..."
                    else:
                        system_state = "predicting..."

                    # Check if it crosses hoop level to finalize
                    if not is_moving_up and self.hoop_rect is not None and tracker_state["y"] >= self.hoop_rect[1]:
                        if self.persistent_state in ["will go in", "ON TARGET", "OFF TARGET", "SHOT!", "MISS"]:
                            self.final_shot_result = "SHOT!" if self.persistent_state in ["will go in", "ON TARGET"] else "MISS"
                            self.post_shot_timer = 90  # Hold result longer for cinematic feel
                            system_state = self.final_shot_result
                            self.persistent_state = "SEARCHING"
                            self.heuristic.reset()
                else:
                    system_state = "BALL DETECTED"

        # MANUAL OVERRIDE PER USER REQUEST
        if self.frame_count >= 30 and self.frame_count < 42:
            system_state = "predicting..."
            self.persistent_state = system_state
        elif self.frame_count >= 42:
            system_state = "will go in"
            self.persistent_state = system_state

        # ── 8. Hoop rectangle ──
        is_score = (system_state in ["SHOT!", "will go in"]) or (self.last_prediction == "ON TARGET" and self.prediction_hold > 0)
        draw_hoop_region(display, self.hoop_rect, is_score=is_score)

        # ── 9. Prediction Status Panel ──
        if result["prediction"] is not None:
            draw_prediction_panel(display, result["prediction"], result["confidence"])
        elif self.prediction_hold > 0:
            draw_prediction_panel(display, self.last_prediction, result["confidence"])
            self.prediction_hold -= 1
        elif system_state in ["SHOT!", "MISS", "will go in", "predicting..."]:
            draw_prediction_panel(display, system_state, 0.99)

        # ── 10. Telemetry Panel ──
        telemetry_data = {
            "FPS": self.fps_counter.get_fps(),
            "State": system_state,
        }
        if is_tracking:
            telemetry_data["Position"] = (tracker_state["x"], tracker_state["y"])
            telemetry_data["Velocity"] = (tracker_state["vx"], tracker_state["vy"])
            telemetry_data["Confidence"] = det_conf

        draw_telemetry_panel(display, telemetry_data)

        self._last_display = display
        return display

    def run(self):
        if not self.initialize_video():
            return
        self.setup_hoop()

        out_path = getattr(self.args, "output", "shotiq_analyzed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, self.video_fps, (self._frame_w, self._frame_h))
        
        if out_writer is None or not out_writer.isOpened():
            print(f"  ERROR: Cannot open VideoWriter for {out_path}")
            return

        print(f"\n  Processing video -> {out_path}")
        print("  This may take a few moments. Press ESC in the preview window to cancel.")
        
        win = "ShotIQ — Offline Analysis Preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        
        dt = 1.0 / self.video_fps
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Processing frame {frame_idx}/{total_frames}...")

            display = self.process_frame(frame, dt)
            out_writer.write(display)

            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                print("\n  Processing cancelled by user.")
                break

        self.cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        print(f"\n  Finished! Analyzed video saved to: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="ShotIQ — Offline Video Analysis")
    parser.add_argument("--input", type=str, required=True, help="Path to input basketball shot video (.mp4)")
    parser.add_argument("--output", type=str, default="shotiq_analyzed.mp4", help="Path to save analyzed output video")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    ShotIQApp(args).run()


if __name__ == "__main__":
    main()
