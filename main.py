"""
ShotIQ — Real-Time Basketball Shot Outcome Prediction System

Main application loop. Orchestrates the detection, tracking,
prediction, and visualization pipeline.

Usage:
    python main.py              — Run with webcam
    python main.py --calibrate  — Enter hoop calibration mode
    python main.py --debug      — Show debug windows (HSV mask, etc.)

Controls:
    ESC / Q  — Quit
    C        — Enter hoop calibration mode
    R        — Reset tracker
    D        — Toggle debug view
    S        — Screenshot current frame
    SPACE    — Pause/Resume

Author: ShotIQ Engineering
"""

import sys
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
    draw_ball_bbox,
    draw_trajectory_trail,
    draw_predicted_trajectory,
    draw_hoop_region,
    draw_prediction_label,
    draw_fps,
    draw_status_bar,
    draw_no_ball_indicator,
)


# ─────────────────────────────────────────────
# Hoop Calibration
# ─────────────────────────────────────────────
class HoopCalibrator:
    """
    Interactive hoop region selector.
    
    User clicks and drags to define the hoop rectangle on a live frame.
    """

    def __init__(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.rect = None
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            self.rect = (x1, y1, x2 - x1, y2 - y1)
            self.done = True

    def calibrate(self, cap):
        """
        Run the calibration UI and return the hoop rectangle.
        
        Returns:
            (x, y, w, h) tuple for the hoop region.
        """
        window_name = "ShotIQ — Click & Drag to Select Hoop Region"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n╔══════════════════════════════════════════╗")
        print("║   HOOP CALIBRATION MODE                  ║")
        print("║   Click and drag to select the hoop area  ║")
        print("║   Press ENTER to confirm, ESC to cancel   ║")
        print("╚══════════════════════════════════════════╝\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            # Draw the selection rectangle
            if self.start_point and self.end_point:
                cv2.rectangle(display, self.start_point, self.end_point,
                            config.COLOR_HOOP, 2)

                # Draw crosshair at center
                cx = (self.start_point[0] + self.end_point[0]) // 2
                cy = (self.start_point[1] + self.end_point[1]) // 2
                cv2.drawMarker(display, (cx, cy), config.COLOR_HOOP,
                             cv2.MARKER_CROSS, 20, 1)

            # Instructions overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            cv2.putText(display, "Drag to select hoop region | ENTER=Confirm | ESC=Cancel",
                       (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                       (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return None
            elif key == 13 and self.rect:  # ENTER
                cv2.destroyWindow(window_name)
                print(f"  ✓ Hoop region set: x={self.rect[0]}, y={self.rect[1]}, "
                      f"w={self.rect[2]}, h={self.rect[3]}")
                return self.rect

        cv2.destroyWindow(window_name)
        return None


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
class ShotIQApp:
    """
    Main application class that orchestrates the entire pipeline.
    """

    def __init__(self, args):
        self.args = args
        self.cap = None
        self.detector = BallDetector()
        self.tracker = BallTracker()
        self.predictor = TrajectoryPredictor()
        self.fps_counter = FPSCounter()

        # Hoop region (x, y, w, h)
        self.hoop_rect = None

        # Application state
        self.paused = False
        self.debug_mode = args.debug if hasattr(args, 'debug') else False
        self.frame_count = 0
        self.shot_count = 0
        self.score_count = 0
        self.last_prediction = None
        self.prediction_hold_frames = 0  # Hold prediction on screen

    def initialize_camera(self):
        """Set up the webcam capture."""
        print("\n🏀 ShotIQ — Basketball Shot Prediction System")
        print("━" * 50)
        print(f"  Camera index: {config.CAMERA_INDEX}")
        print(f"  Target resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print(f"  Target FPS: {config.FPS_TARGET}")
        print("━" * 50)

        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if not self.cap.isOpened():
            print("\n❌ ERROR: Could not open camera!")
            print("   Try changing CAMERA_INDEX in config.py")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS_TARGET)

        # Read actual values (camera might not support requested settings)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"\n  ✓ Camera opened successfully")
        print(f"  ✓ Actual resolution: {actual_w}x{actual_h}")
        print(f"  ✓ Actual FPS: {actual_fps:.0f}\n")

        return True

    def setup_hoop(self):
        """Initialize or calibrate the hoop region."""
        if self.args.calibrate or (config.HOOP_X is None):
            print("  ℹ  No hoop position configured — entering calibration mode...")
            calibrator = HoopCalibrator()
            self.hoop_rect = calibrator.calibrate(self.cap)

            if self.hoop_rect is None:
                # Use a default center-right position
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.hoop_rect = (
                    actual_w - 200,
                    actual_h // 3,
                    config.HOOP_WIDTH,
                    config.HOOP_HEIGHT
                )
                print(f"  ⚠  Using default hoop position: {self.hoop_rect}")
        else:
            self.hoop_rect = (
                config.HOOP_X,
                config.HOOP_Y,
                config.HOOP_WIDTH,
                config.HOOP_HEIGHT
            )
            print(f"  ✓ Hoop position loaded from config: {self.hoop_rect}")

    def process_frame(self, frame):
        """
        Run the full detection → tracking → prediction pipeline on one frame.

        Returns:
            The annotated frame ready for display.
        """
        self.frame_count += 1
        display = frame.copy()

        # ── Step 1: Detect the ball ──
        detection = self.detector.detect(frame)

        # ── Step 2: Update tracker ──
        is_tracking = self.tracker.update(detection)

        # ── Step 3: Draw ball bounding box ──
        if detection is not None:
            draw_ball_bbox(
                display,
                detection["x"], detection["y"],
                detection["w"], detection["h"]
            )

        # ── Step 4: Draw trajectory trail ──
        smoothed_positions = self.tracker.get_smoothed_positions()
        if len(smoothed_positions) > 1:
            draw_trajectory_trail(display, smoothed_positions)

        # ── Step 5: Predict trajectory & score ──
        result = {"prediction": None, "confidence": 0, "predicted_points": []}

        if self.tracker.has_enough_data() and self.hoop_rect is not None:
            raw_positions = self.tracker.get_raw_positions()
            result = self.predictor.predict(raw_positions, self.hoop_rect)

            # Draw predicted trajectory
            if result["predicted_points"]:
                draw_predicted_trajectory(display, result["predicted_points"])

            # Update prediction state
            if result["prediction"] is not None:
                self.last_prediction = result["prediction"]
                self.prediction_hold_frames = 45  # Hold for ~1.5 seconds at 30fps

        # ── Step 6: Draw hoop region ──
        if self.hoop_rect is not None:
            is_score = (result["prediction"] == "SCORE") or \
                       (self.last_prediction == "SCORE" and self.prediction_hold_frames > 0)
            draw_hoop_region(display, self.hoop_rect, is_score=is_score)

        # ── Step 7: Draw prediction label ──
        frame_w = frame.shape[1]

        if result["prediction"] is not None:
            draw_prediction_label(display, result["prediction"],
                                result["confidence"], frame_w)
        elif self.prediction_hold_frames > 0:
            # Show the last prediction for a bit even after ball is lost
            hold_confidence = result.get("confidence", 0.5)
            draw_prediction_label(display, self.last_prediction,
                                hold_confidence, frame_w)
            self.prediction_hold_frames -= 1

        # ── Step 8: Draw "no ball" indicator ──
        if not is_tracking and detection is None:
            draw_no_ball_indicator(display, self.tracker.get_frames_missing())

        # ── Step 9: FPS counter ──
        self.fps_counter.tick()
        draw_fps(display, self.fps_counter.get_fps())

        # ── Step 10: Status bar ──
        speed = self.tracker.get_speed()
        n_points = len(self.tracker.get_positions())
        status = (f"Tracking: {'ON' if is_tracking else 'OFF'} | "
                  f"Points: {n_points} | "
                  f"Speed: {speed:.0f} px/f | "
                  f"Frame: {self.frame_count}")
        draw_status_bar(display, status, frame.shape[0], frame.shape[1])

        return display

    def run(self):
        """Main application loop."""
        if not self.initialize_camera():
            return

        self.setup_hoop()

        window_name = "ShotIQ — Basketball Shot Prediction"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n  Controls:")
        print("  ─────────────────────────────")
        print("  ESC/Q  — Quit")
        print("  C      — Recalibrate hoop")
        print("  R      — Reset tracker")
        print("  D      — Toggle debug view")
        print("  S      — Save screenshot")
        print("  SPACE  — Pause / Resume")
        print("  ─────────────────────────────\n")
        print("  🎬 Running... Press ESC to quit.\n")

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("  ❌ Frame capture failed. Retrying...")
                    continue

                # Process the frame through our pipeline
                display = self.process_frame(frame)

                # Debug view: show the HSV mask in a second window
                if self.debug_mode:
                    mask = self.detector.get_debug_mask()
                    if mask is not None:
                        # Create a colored overlay of the mask
                        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        mask_colored[:, :, 0] = 0   # Remove blue channel
                        mask_colored[:, :, 2] = 0   # Remove red channel — shows green

                        # Stack original and mask side by side
                        h, w = frame.shape[:2]
                        mask_resized = cv2.resize(mask_colored, (w, h))
                        debug_view = np.hstack([frame, mask_resized])
                        debug_view = cv2.resize(debug_view, (w, h // 2 * 2))
                        cv2.imshow("ShotIQ — Debug: HSV Mask", debug_view)

                cv2.imshow(window_name, display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC or Q
                break

            elif key == ord('c'):  # Recalibrate hoop
                calibrator = HoopCalibrator()
                new_rect = calibrator.calibrate(self.cap)
                if new_rect is not None:
                    self.hoop_rect = new_rect

            elif key == ord('r'):  # Reset tracker
                self.tracker.reset()
                self.last_prediction = None
                self.prediction_hold_frames = 0
                print("  ↺ Tracker reset")

            elif key == ord('d'):  # Toggle debug
                self.debug_mode = not self.debug_mode
                if not self.debug_mode:
                    cv2.destroyWindow("ShotIQ — Debug: HSV Mask")
                print(f"  Debug mode: {'ON' if self.debug_mode else 'OFF'}")

            elif key == ord('s'):  # Screenshot
                filename = f"shotiq_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                print(f"  📸 Screenshot saved: {filename}")

            elif key == 32:  # SPACE — pause
                self.paused = not self.paused
                print(f"  {'⏸ Paused' if self.paused else '▶ Resumed'}")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n  ✓ ShotIQ shut down cleanly.\n")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ShotIQ — Real-Time Basketball Shot Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 Run with default webcam
  python main.py --calibrate     Force hoop calibration on start
  python main.py --debug         Enable debug visualization
  python main.py --camera 1      Use camera index 1
        """
    )
    parser.add_argument("--calibrate", action="store_true",
                        help="Force hoop calibration mode on startup")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug windows (HSV mask)")
    parser.add_argument("--camera", type=int, default=None,
                        help="Override camera index")

    args = parser.parse_args()

    # Override camera index if provided
    if args.camera is not None:
        config.CAMERA_INDEX = args.camera

    app = ShotIQApp(args)
    app.run()


if __name__ == "__main__":
    main()
