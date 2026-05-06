"""
Body & Hand Tracking Module for the Basketball Shot Prediction System.

Uses MediaPipe Pose and Hands to:
  - Track full body skeleton (33 landmarks)
  - Track hand landmarks (21 per hand)
  - Detect shooting arm angles and form
  - Determine ball-holding state
  - Identify shooting phases (set, release, follow-through)
  - Provide wrist/fingertip positions for ball-hand association

Dependencies: mediapipe, opencv-python, numpy
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import config


class BodyTracker:
    """
    Tracks body pose and hand landmarks using MediaPipe.

    Provides:
      - Full body skeleton with 33 pose landmarks
      - Hand landmarks (21 per hand, up to 2 hands)
      - Shooting arm detection and angle computation
      - Ball-in-hand proximity detection
      - Shooting phase classification
    """

    def __init__(self):
        # ── MediaPipe Pose ──
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=config.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE,
        )

        # ── State ──
        self.pose_landmarks = None
        self.hand_landmarks_list = []
        self.handedness_list = []

        # ── Shooting analysis ──
        self.shooting_arm = None       # "left" or "right"
        self.elbow_angle = 0.0
        self.shoulder_angle = 0.0
        self.wrist_positions = deque(maxlen=30)
        self.shooting_phase = "IDLE"   # IDLE, SET, RELEASE, FOLLOW_THROUGH
        self.release_detected = False
        self.ball_in_hand = False

        # ── Frame dimensions (set on first process) ──
        self.frame_w = 0
        self.frame_h = 0

        # ── Wrist velocity tracking for release detection ──
        self.wrist_history = deque(maxlen=10)
        self.prev_wrist_y = None

        # ── Key landmark pixel positions (updated each frame) ──
        self.key_points = {}

    # ─────────────────────────────────────────────
    # Main Processing
    # ─────────────────────────────────────────────
    def process(self, frame):
        """
        Run pose and hand detection on a single frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            dict with pose and hand data for downstream use.
        """
        self.frame_h, self.frame_w = frame.shape[:2]

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # ── Run Pose detection ──
        pose_results = self.pose.process(rgb_frame)
        self.pose_landmarks = pose_results.pose_landmarks

        # ── Run Hand detection ──
        hand_results = self.hands.process(rgb_frame)
        self.hand_landmarks_list = hand_results.multi_hand_landmarks or []
        self.handedness_list = hand_results.multi_handedness or []

        rgb_frame.flags.writeable = True

        # ── Extract key points in pixel coordinates ──
        self._extract_key_points()

        # ── Analyze shooting form ──
        self._analyze_shooting_form()

        # ── Detect shooting phase ──
        self._detect_shooting_phase()

        return self._get_result()

    def _extract_key_points(self):
        """Extract key body landmarks as pixel coordinates."""
        self.key_points = {}

        if self.pose_landmarks is None:
            return

        landmarks = self.pose_landmarks.landmark

        # Map important landmarks to pixel coords
        point_names = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_index': self.mp_pose.PoseLandmark.LEFT_INDEX,
            'right_index': self.mp_pose.PoseLandmark.RIGHT_INDEX,
            'left_pinky': self.mp_pose.PoseLandmark.LEFT_PINKY,
            'right_pinky': self.mp_pose.PoseLandmark.RIGHT_PINKY,
        }

        for name, landmark_id in point_names.items():
            lm = landmarks[landmark_id]
            if lm.visibility > config.POSE_VISIBILITY_THRESHOLD:
                px = int(lm.x * self.frame_w)
                py = int(lm.y * self.frame_h)
                self.key_points[name] = (px, py, lm.visibility)

    def _analyze_shooting_form(self):
        """Analyze the shooting arm position and angles."""
        if self.pose_landmarks is None:
            return

        # ── Determine dominant/shooting arm ──
        # The shooting arm is typically the one raised higher
        left_wrist = self.key_points.get('left_wrist')
        right_wrist = self.key_points.get('right_wrist')
        left_shoulder = self.key_points.get('left_shoulder')
        right_shoulder = self.key_points.get('right_shoulder')

        if left_wrist and right_wrist:
            # The arm with the higher wrist (lower y in pixel coords) is likely shooting
            if left_wrist[1] < right_wrist[1]:
                self.shooting_arm = "left"
            else:
                self.shooting_arm = "right"
        elif left_wrist:
            self.shooting_arm = "left"
        elif right_wrist:
            self.shooting_arm = "right"

        # ── Compute elbow angle for the shooting arm ──
        if self.shooting_arm == "right":
            shoulder = self.key_points.get('right_shoulder')
            elbow = self.key_points.get('right_elbow')
            wrist = self.key_points.get('right_wrist')
        else:
            shoulder = self.key_points.get('left_shoulder')
            elbow = self.key_points.get('left_elbow')
            wrist = self.key_points.get('left_wrist')

        if shoulder and elbow and wrist:
            self.elbow_angle = self._compute_angle(
                shoulder[:2], elbow[:2], wrist[:2]
            )
            self.shoulder_angle = self._compute_angle(
                self.key_points.get('right_hip', self.key_points.get('left_hip', (0, 0, 0)))[:2],
                shoulder[:2], elbow[:2]
            )

        # ── Track wrist position ──
        wrist_pos = self.key_points.get(
            f'{self.shooting_arm}_wrist' if self.shooting_arm else 'right_wrist'
        )
        if wrist_pos:
            self.wrist_positions.append(wrist_pos[:2])
            self.wrist_history.append(wrist_pos[:2])

    def _detect_shooting_phase(self):
        """
        Classify the current shooting phase based on body pose.

        Phases:
          IDLE          — Standing, no shot motion
          SET           — Ball raised, elbow bent (preparation)
          RELEASE       — Rapid upward wrist motion (shooting)
          FOLLOW_THROUGH — Arm extended after release
        """
        if len(self.wrist_history) < 3 or self.pose_landmarks is None:
            self.shooting_phase = "IDLE"
            return

        # Compute wrist vertical velocity (pixels/frame, negative = moving up)
        recent_wrist = list(self.wrist_history)
        vy_wrist = recent_wrist[-1][1] - recent_wrist[-3][1]  # y delta over 3 frames

        # Shooting arm raised above shoulder?
        wrist = self.key_points.get(
            f'{self.shooting_arm}_wrist' if self.shooting_arm else 'right_wrist'
        )
        shoulder = self.key_points.get(
            f'{self.shooting_arm}_shoulder' if self.shooting_arm else 'right_shoulder'
        )

        arm_raised = False
        if wrist and shoulder:
            arm_raised = wrist[1] < shoulder[1]  # wrist above shoulder

        # ── Phase classification ──
        prev_phase = self.shooting_phase

        if self.elbow_angle > 150 and arm_raised:
            # Arm extended, above shoulder = follow-through
            self.shooting_phase = "FOLLOW_THROUGH"
        elif vy_wrist < -config.RELEASE_VELOCITY_THRESHOLD and arm_raised:
            # Rapid upward wrist motion = release
            self.shooting_phase = "RELEASE"
            self.release_detected = True
        elif 60 < self.elbow_angle < 130 and arm_raised:
            # Elbow bent, arm raised = set position
            self.shooting_phase = "SET"
        else:
            self.shooting_phase = "IDLE"
            if prev_phase == "FOLLOW_THROUGH":
                self.release_detected = False

    # ─────────────────────────────────────────────
    # Ball-Hand Association
    # ─────────────────────────────────────────────
    def check_ball_proximity(self, ball_position):
        """
        Check if the ball is near either hand (within grasp distance).

        Args:
            ball_position: (x, y) of the detected ball center.

        Returns:
            dict with proximity info:
              - in_hand: bool
              - hand: "left" or "right" or None
              - distance: float (pixels)
        """
        if ball_position is None:
            self.ball_in_hand = False
            return {"in_hand": False, "hand": None, "distance": float('inf')}

        bx, by = ball_position
        min_dist = float('inf')
        closest_hand = None

        # Check against pose wrist positions
        for side in ['left', 'right']:
            wrist = self.key_points.get(f'{side}_wrist')
            if wrist:
                dist = np.sqrt((bx - wrist[0])**2 + (by - wrist[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_hand = side

            # Also check index fingertip from pose
            index = self.key_points.get(f'{side}_index')
            if index:
                dist = np.sqrt((bx - index[0])**2 + (by - index[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_hand = side

        # Check against hand landmark fingertips (more precise)
        for i, hand_lms in enumerate(self.hand_landmarks_list):
            # Get fingertip landmarks (indices 4, 8, 12, 16, 20)
            fingertip_indices = [4, 8, 12, 16, 20]
            for tip_idx in fingertip_indices:
                tip = hand_lms.landmark[tip_idx]
                tx = int(tip.x * self.frame_w)
                ty = int(tip.y * self.frame_h)
                dist = np.sqrt((bx - tx)**2 + (by - ty)**2)
                if dist < min_dist:
                    min_dist = dist
                    # Determine hand side from handedness
                    if i < len(self.handedness_list):
                        label = self.handedness_list[i].classification[0].label
                        closest_hand = label.lower()

            # Also check palm center (landmark 9)
            palm = hand_lms.landmark[9]
            px = int(palm.x * self.frame_w)
            py = int(palm.y * self.frame_h)
            dist = np.sqrt((bx - px)**2 + (by - py)**2)
            if dist < min_dist:
                min_dist = dist

        self.ball_in_hand = min_dist < config.BALL_HAND_PROXIMITY_THRESHOLD
        return {
            "in_hand": self.ball_in_hand,
            "hand": closest_hand,
            "distance": min_dist
        }

    def get_hand_fingertip_positions(self):
        """
        Get all fingertip positions from hand landmarks.

        Returns:
            List of (x, y) tuples for all detected fingertips.
        """
        fingertips = []
        fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips

        for hand_lms in self.hand_landmarks_list:
            for tip_idx in fingertip_indices:
                tip = hand_lms.landmark[tip_idx]
                fx = int(tip.x * self.frame_w)
                fy = int(tip.y * self.frame_h)
                fingertips.append((fx, fy))

        return fingertips

    def get_wrist_positions(self):
        """Get wrist positions from pose landmarks."""
        wrists = {}
        for side in ['left', 'right']:
            wrist = self.key_points.get(f'{side}_wrist')
            if wrist:
                wrists[side] = wrist[:2]
        return wrists

    # ─────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────
    def draw_skeleton(self, frame):
        """Draw the full body skeleton with custom styling."""
        if self.pose_landmarks is None:
            return frame

        # Draw pose connections with custom colors
        self.mp_drawing.draw_landmarks(
            frame,
            self.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=config.COLOR_POSE_LANDMARKS,
                thickness=2,
                circle_radius=3,
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=config.COLOR_POSE_CONNECTIONS,
                thickness=2,
            ),
        )

        # Highlight shooting arm with a different color
        if self.shooting_arm:
            self._draw_shooting_arm_highlight(frame)

        return frame

    def _draw_shooting_arm_highlight(self, frame):
        """Highlight the detected shooting arm with accent color."""
        if self.shooting_arm == "right":
            shoulder = self.key_points.get('right_shoulder')
            elbow = self.key_points.get('right_elbow')
            wrist = self.key_points.get('right_wrist')
        else:
            shoulder = self.key_points.get('left_shoulder')
            elbow = self.key_points.get('left_elbow')
            wrist = self.key_points.get('left_wrist')

        if shoulder and elbow:
            cv2.line(frame, shoulder[:2], elbow[:2],
                     config.COLOR_SHOOTING_ARM, 3, cv2.LINE_AA)
        if elbow and wrist:
            cv2.line(frame, elbow[:2], wrist[:2],
                     config.COLOR_SHOOTING_ARM, 3, cv2.LINE_AA)

        # Draw elbow angle arc
        if shoulder and elbow and wrist:
            self._draw_angle_arc(frame, shoulder[:2], elbow[:2], wrist[:2],
                                 self.elbow_angle)

    def _draw_angle_arc(self, frame, p1, vertex, p2, angle):
        """Draw an angle arc at a joint."""
        # Vectors from vertex to p1 and p2
        v1 = np.array(p1) - np.array(vertex)
        v2 = np.array(p2) - np.array(vertex)

        # Starting angle
        start_angle = np.degrees(np.arctan2(-v1[1], v1[0]))
        end_angle = np.degrees(np.arctan2(-v2[1], v2[0]))

        # Draw the arc
        radius = 25
        color = config.COLOR_ANGLE_ARC
        if angle < 90:
            color = (0, 100, 255)  # Orange-ish for tight angle
        elif angle > 150:
            color = (0, 255, 100)  # Green for extended

        cv2.ellipse(frame, vertex, (radius, radius),
                     0, -start_angle, -end_angle, color, 2, cv2.LINE_AA)

        # Draw angle text
        mid_x = vertex[0] + int(35 * np.cos(np.radians(-(start_angle + end_angle) / 2)))
        mid_y = vertex[1] + int(35 * np.sin(np.radians(-(start_angle + end_angle) / 2)))
        cv2.putText(frame, f"{angle:.0f}°", (mid_x, mid_y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def draw_hands(self, frame):
        """Draw hand landmarks with connections."""
        for hand_lms in self.hand_landmarks_list:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=config.COLOR_HAND_LANDMARKS,
                    thickness=1,
                    circle_radius=2,
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=config.COLOR_HAND_CONNECTIONS,
                    thickness=1,
                ),
            )
        return frame

    def draw_shooting_info(self, frame):
        """Draw shooting phase and form analysis overlay."""
        if self.pose_landmarks is None:
            return frame

        # ── Shooting phase badge ──
        phase_colors = {
            "IDLE": (120, 120, 120),
            "SET": (0, 200, 255),      # Orange
            "RELEASE": (0, 100, 255),  # Red-orange
            "FOLLOW_THROUGH": (0, 255, 100),  # Green
        }
        phase_color = phase_colors.get(self.shooting_phase, (120, 120, 120))

        # Panel background
        panel_x, panel_y = 10, 55
        panel_w, panel_h = 220, 95
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      config.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Phase indicator with colored dot
        cv2.circle(frame, (panel_x + 12, panel_y + 18), 6, phase_color, -1)
        cv2.putText(frame, f"Phase: {self.shooting_phase}",
                    (panel_x + 25, panel_y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1, cv2.LINE_AA)

        # Elbow angle
        cv2.putText(frame, f"Elbow: {self.elbow_angle:.0f}°",
                    (panel_x + 10, panel_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Shooting arm
        arm_text = f"Arm: {self.shooting_arm.upper() if self.shooting_arm else 'N/A'}"
        cv2.putText(frame, arm_text,
                    (panel_x + 10, panel_y + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Ball in hand
        ball_text = "Ball: IN HAND" if self.ball_in_hand else "Ball: FREE"
        ball_color = (0, 255, 100) if self.ball_in_hand else (150, 150, 150)
        cv2.putText(frame, ball_text,
                    (panel_x + 10, panel_y + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ball_color, 1, cv2.LINE_AA)

        return frame

    def draw_ball_hand_link(self, frame, ball_position):
        """Draw a line between the ball and the closest hand when close."""
        if ball_position is None or not self.ball_in_hand:
            return frame

        wrists = self.get_wrist_positions()
        if not wrists:
            return frame

        # Find closest wrist
        bx, by = ball_position
        min_dist = float('inf')
        closest_wrist = None
        for side, pos in wrists.items():
            dist = np.sqrt((bx - pos[0])**2 + (by - pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_wrist = pos

        if closest_wrist and min_dist < config.BALL_HAND_PROXIMITY_THRESHOLD:
            # Draw dashed line
            cv2.line(frame, ball_position, closest_wrist,
                     config.COLOR_BALL_HAND_LINK, 1, cv2.LINE_AA)
            # Draw grip indicator circle
            mid = ((bx + closest_wrist[0]) // 2, (by + closest_wrist[1]) // 2)
            cv2.circle(frame, mid, 4, config.COLOR_BALL_HAND_LINK, -1)

        return frame

    # ─────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────
    @staticmethod
    def _compute_angle(p1, p2, p3):
        """
        Compute the angle at p2 formed by segments p1-p2 and p2-p3.

        Returns:
            Angle in degrees (0 - 180).
        """
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    def _get_result(self):
        """Package the body tracking results."""
        return {
            "pose_detected": self.pose_landmarks is not None,
            "hands_detected": len(self.hand_landmarks_list) > 0,
            "num_hands": len(self.hand_landmarks_list),
            "shooting_arm": self.shooting_arm,
            "elbow_angle": self.elbow_angle,
            "shooting_phase": self.shooting_phase,
            "ball_in_hand": self.ball_in_hand,
            "key_points": self.key_points,
        }

    def get_shooting_phase(self):
        """Return the current shooting phase."""
        return self.shooting_phase

    def get_elbow_angle(self):
        """Return the shooting arm elbow angle."""
        return self.elbow_angle

    def is_release_detected(self):
        """Return True if a shot release was just detected."""
        return self.release_detected

    def release(self):
        """Release MediaPipe resources."""
        self.pose.close()
        self.hands.close()
