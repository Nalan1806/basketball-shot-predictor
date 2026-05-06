"""
Extended Kalman Filter (EKF) Ball Tracker for ShotIQ.

5-state EKF: [px, py, vx, vy, beta]
  px, py  — world position (metres)
  vx, vy  — world velocity (m/s)
  beta    — aerodynamic drag correction factor

Key stability features:
  - lost_frames counter: 0 = real detection, 1–10 = predict-only with bounds clamping,
    > 10 = full hard reset
  - Covariance trace threshold: trace(P) > 1000 → full reset
  - covariance_healthy() gate: trace(P) <= 200 AND lost_frames == 0 (strict filter for drawing)
  - Frame-bound clamping: if predicted pixel position leaves frame → reset immediately
  - Velocity clamping in state transition
  - Joseph-form covariance update for numerical symmetry
  - Per-step sanitize_state() clamps all state components

Public tracking methods:
  update(detection, confidence)      — main per-frame update
  predict_trajectory(n_steps)        — EKF forward projection (pixels)
  get_uncertainty_ellipse(step)      — 66% confidence ellipse (only if healthy)
  covariance_healthy()               — True if trace(P) <= 200 AND lost_frames == 0
  get_lost_frames()                  — current lost_frames count
"""

import numpy as np
from pixel_to_world import PixelToWorld

# ─────────────────────────────────────────────
# Physics constants
# ─────────────────────────────────────────────
GRAVITY       = 9.81
BALL_MASS     = 0.623
BALL_RADIUS   = 0.12
AIR_DENSITY   = 1.2
DRAG_COEFF_CD = 0.47

_BALL_AREA  = np.pi * BALL_RADIUS ** 2
ALPHA_DRAG  = -DRAG_COEFF_CD * _BALL_AREA * AIR_DENSITY / (2.0 * BALL_MASS)

DT = 1.0 / 30.0   # seconds per frame (30 FPS)

# ─────────────────────────────────────────────
# EKF noise / covariance parameters
# ─────────────────────────────────────────────
Q_POS     = 0.001
Q_VEL     = 0.10
Q_BETA    = 0.001
R_POS     = 0.05

P0_POS    = 0.5
P0_VEL    = 2.0
P0_BETA   = 0.25

P_MAX_DIAG      = 1e4        # per-element cap
MAX_COV_TRACE   = 1000.0     # trace(P) above this → reset EKF

BETA_INIT = 1.0

# ─────────────────────────────────────────────
# Tracking limits — lost_frames based
# ─────────────────────────────────────────────
FORWARD_STEPS   = 200        # future projection steps
COAST_FRAMES    = 10         # lost_frames: predict-only before hard reset (> 10 = reset)
MAX_COAST_DIST  = 300        # max jump allowed during coasting (pixels)

# Basket geometry (metres)
BASKET_HEIGHT_M  = 3.05
BASKET_X_NEAR    = 4.57 - 0.46
BASKET_X_FAR     = 4.57
RING_TOLERANCE_M = 0.08

CHI2_66_2DOF = 1.8304     # sqrt(chi2inv(0.66, 2)), no scipy needed

# ─────────────────────────────────────────────
# Frame bounds (pixels) — set by tracker when frame size is known
# ─────────────────────────────────────────────
_FRAME_W = 1280
_FRAME_H = 720


class EKFBallTracker:
    """
    5-state Extended Kalman Filter with aerodynamic drag physics,
    automatic stability resets, and lost_frames-based state management.

    Lost frames logic:
      0         → real detection received, reset counter
      1–8       → predict-only mode, clamp predicted position to frame bounds
      > 8       → hard reset (clear state + trajectory history)
    """

    def __init__(self, pixel_to_world=None, frame_w=1280, frame_h=720):
        """
        Initialize the EKF tracker.

        Args:
            pixel_to_world: PixelToWorld instance for coordinate transforms.
            frame_w, frame_h: frame dimensions in pixels.
        """
        global _FRAME_W, _FRAME_H
        _FRAME_W, _FRAME_H = frame_w, frame_h

        self.p2w = pixel_to_world if pixel_to_world is not None else PixelToWorld()

        # ── EKF state ──
        self.x = np.zeros((5, 1), dtype=np.float64)
        self.P = np.eye(5, dtype=np.float64)
        self.Q = np.diag([Q_POS, Q_POS, Q_VEL, Q_VEL, Q_BETA])
        self.R = np.eye(2, dtype=np.float64) * R_POS
        self.H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float64)

        self.initialized = False

        # ── Tracking buffers ──
        from collections import deque
        import config
        max_pts       = getattr(config, 'MAX_TRACK_POINTS', 50)
        self._min_pts = getattr(config, 'MIN_POINTS_FOR_PREDICTION', 6)
        self._max_dist_jump = getattr(config, 'MAX_DISTANCE_JUMP', 300)

        self.positions       = deque(maxlen=max_pts)
        self.raw_positions   = deque(maxlen=max_pts)
        self._world_positions = []
        self._future_positions = []
        self._future_covs      = []

        # ── Tracking state ──
        self.frames_missing     = 0
        self.lost_frames        = 0       # NEW: frames with no real detection
        self.is_tracking        = False
        self.last_position      = None
        self.last_detection_pos = None
        self.predicted_position = None
        self.velocity           = (0.0, 0.0)
        self.current_confidence = 0.0
        self.consecutive_good   = 0

        import config as _cfg
        self._max_frames_missing = getattr(_cfg, 'MAX_FRAMES_MISSING', COAST_FRAMES)

    # ──────────────────────────────────────────────────────────────────
    # Physics / State Transition
    # ──────────────────────────────────────────────────────────────────

    def _f(self, x, dt=DT):
        """
        Euler state transition with velocity clamping.

        Integrates drag physics:
          - Drag acceleration proportional to speed * velocity * beta factor
          - Gravity always acts downward
          - Velocity clamped to ±50 m/s to prevent divergence

        Args:
            x: (5, 1) state vector [px, py, vx, vy, beta]
            dt: time step (seconds)

        Returns:
            (5, 1) new state
        """
        V_MAX = 50.0
        px, py, vx, vy, beta = x.flatten()

        speed = np.hypot(vx, vy) + 1e-9

        drag_x = ALPHA_DRAG * beta * speed * vx
        drag_y = ALPHA_DRAG * beta * speed * vy

        return np.array([
            [px + vx * dt],
            [py + vy * dt],
            [float(np.clip(vx + drag_x * dt, -V_MAX, V_MAX))],
            [float(np.clip(vy + (drag_y + GRAVITY) * dt, -V_MAX, V_MAX))],
            [float(np.clip(beta, 0.1, 5.0))],
        ])

    def _jacobian_F(self, x, dt=DT):
        """
        5×5 Jacobian matrix of state transition for EKF predict step.

        Linearization around current state for covariance propagation.

        Args:
            x: (5, 1) current state
            dt: time step (seconds)

        Returns:
            (5, 5) Jacobian matrix
        """
        _, _, vx, vy, beta = x.flatten()

        speed = np.hypot(vx, vy) + 1e-9

        ab = ALPHA_DRAG * beta

        # Drag gradient w.r.t. velocity
        ddx_dvx = ab * (speed + vx ** 2 / speed)
        ddx_dvy = ab * (vx * vy / speed)
        ddx_db = ALPHA_DRAG * speed * vx

        ddy_dvx = ab * (vx * vy / speed)
        ddy_dvy = ab * (speed + vy ** 2 / speed)
        ddy_db = ALPHA_DRAG * speed * vy

        # Initialize as identity
        F = np.eye(5, dtype=np.float64)

        # Position derivatives (x += vx*dt, y += vy*dt)
        F[0, 2] = dt
        F[1, 3] = dt

        # Velocity derivatives (w.r.t. drag + gravity)
        F[2, 2] = 1.0 + ddx_dvx * dt
        F[2, 3] = ddx_dvy * dt
        F[2, 4] = ddx_db * dt

        F[3, 2] = ddy_dvx * dt
        F[3, 3] = 1.0 + ddy_dvy * dt
        F[3, 4] = ddy_db * dt

        return F

    # ──────────────────────────────────────────────────────────────────
    # EKF Core Steps
    # ──────────────────────────────────────────────────────────────────

    def _init_state(self, wx, wy, vx=0.0, vy=0.0):
        """
        Initialize the EKF state.

        Args:
            wx, wy: initial position in world coordinates (metres)
            vx, vy: initial velocity (m/s)
        """
        self.x = np.array([[wx], [wy], [vx], [vy], [BETA_INIT]])
        self.P = np.diag([P0_POS, P0_POS, P0_VEL, P0_VEL, P0_BETA]).astype(np.float64)
        self.initialized = True

    def _sanitize_P(self, P):
        """
        Ensure covariance matrix P remains valid and well-conditioned.

        Operations:
          - Symmetrize (P = 0.5*(P + P.T))
          - Clamp diagonal elements to [0, P_MAX_DIAG]
          - Guard against NaN values

        Args:
            P: (5, 5) covariance matrix

        Returns:
            (5, 5) cleaned covariance matrix
        """
        P = 0.5 * (P + P.T)
        np.clip(np.diag(P), 0, P_MAX_DIAG, out=P.flat[::6])
        if not np.isfinite(P).all():
            P = np.diag([P0_POS, P0_POS, P0_VEL, P0_VEL, P0_BETA])
        return P

    def _sanitize_state(self, x_prev=None):
        """
        Clamp state to physical bounds; recover from NaN.

        Ensures:
          - Position stays within ±50m (x), ±20m (y)
          - Velocity stays within ±50 m/s (both axes)
          - Beta (drag) stays within [0.1, 5.0]
          - No NaN or Inf values

        Args:
            x_prev: previous state for NaN recovery (optional)
        """
        if not np.isfinite(self.x).all():
            # Recover from NaN
            if x_prev is not None:
                self.x[0, 0] = float(x_prev[0, 0])
                self.x[1, 0] = float(x_prev[1, 0])
            else:
                self.x[0, 0] = self.x[1, 0] = 0.0
            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0
            self.x[4, 0] = BETA_INIT
            return

        # Clamp all state components
        self.x[0, 0] = float(np.clip(self.x[0, 0], -50.0, 50.0))
        self.x[1, 0] = float(np.clip(self.x[1, 0], -20.0, 20.0))
        self.x[2, 0] = float(np.clip(self.x[2, 0], -50.0, 50.0))
        self.x[3, 0] = float(np.clip(self.x[3, 0], -50.0, 50.0))
        self.x[4, 0] = float(np.clip(self.x[4, 0], 0.1, 5.0))

    def _ekf_predict(self):
        """
        EKF predict step: propagate state and covariance forward.

        State: x_new = f(x_old)
        Covariance: P_new = F @ P_old @ F^T + Q

        Includes state sanitization and covariance cleaning.
        """
        x_prev = self.x.copy()
        F = self._jacobian_F(self.x)
        self.x = self._f(self.x)
        self.P = self._sanitize_P(F @ self.P @ F.T + self.Q)
        self._sanitize_state(x_prev)

    def _ekf_update(self, z):
        """
        EKF update step: assimilate measurement (real detection).

        Measurement: z = [px_world, py_world]
        Update: x_new = x_old + K @ (z - H @ x_old)
        Covariance update uses Joseph form for numerical stability

        Args:
            z: (2, 1) measurement vector [wx, wy]
        """
        x_prev = self.x.copy()

        # Innovation covariance: S = H @ P @ H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P @ H^T @ inv(S)
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix — skip update
            return

        # State update: x = x + K @ (z - H@x)
        self.x = self.x + K @ (z - self.H @ self.x)

        # Joseph form covariance update: P = (I - K@H) @ P @ (I - K@H)^T + K @ R @ K^T
        # More numerically stable than standard form
        I_KH = np.eye(5) - K @ self.H
        self.P = self._sanitize_P(I_KH @ self.P @ I_KH.T + K @ self.R @ K.T)

        self._sanitize_state(x_prev)

    # ──────────────────────────────────────────────────────────────────
    # Stability Checks
    # ──────────────────────────────────────────────────────────────────

    def covariance_healthy(self):
        """
        Check if EKF covariance is well-conditioned and actively tracking.

        Criteria:
          - trace(P) <= 200.0 (very tight covariance gate)
          - lost_frames == 0 (actively tracking, no coasting)

        Returns:
            bool: True if both conditions met
        """
        trace_healthy = float(np.trace(self.P)) <= 200.0
        tracking_healthy = (self.lost_frames == 0)
        return trace_healthy and tracking_healthy

    def get_lost_frames(self):
        """Return the current lost_frames counter."""
        return self.lost_frames

    def _check_and_reset_if_diverged(self, frame_w=None, frame_h=None):
        """
        Auto-reset tracker if stability thresholds are breached.

        Resets if:
          1. Covariance trace(P) > 1000.0
          2. Predicted pixel position leaves frame bounds

        Args:
            frame_w, frame_h: frame dimensions (optional; uses globals if None)

        Returns:
            bool: True if reset was triggered
        """
        fw = frame_w or _FRAME_W
        fh = frame_h or _FRAME_H

        # Check 1: Covariance trace
        if float(np.trace(self.P)) > 1000.0:
            self.reset()
            return True

        # Check 2: Predicted position outside frame bounds
        if self.initialized:
            try:
                ppx, ppy = self.p2w.to_pixel(float(self.x[0, 0]), float(self.x[1, 0]))

                # Allow 20% frame margin for numerical tolerance
                margin_x = fw * 0.2
                margin_y = fh * 0.2

                if ppx < -margin_x or ppx > fw + margin_x or \
                   ppy < -margin_y or ppy > fh + margin_y:
                    self.reset()
                    return True
            except Exception:
                self.reset()
                return True

        return False

    # ──────────────────────────────────────────────────────────────────
    # Main Update Method (called once per frame)
    # ──────────────────────────────────────────────────────────────────

    def update(self, detection, confidence=0.0):
        """
        Update EKF with latest ball detection (or None to coast).

        Logic:
          - If detection provided:
              * Reset lost_frames to 0 (real detection)
              * Convert pixel → world coordinates
              * Initialize EKF if needed
              * Run EKF predict + update steps
          - If no detection:
              * Increment lost_frames
              * If lost_frames > 10: full reset
              * If 1 <= lost_frames <= 10: predict-only, clamp to bounds
          - Stability checks: trace(P) > 1000 or pixel position outside frame → reset

        Args:
            detection: dict with "cx", "cy" keys, or None
            confidence: float 0–1 detection confidence

        Returns:
            bool: True if actively tracking, False otherwise
        """
        self.current_confidence = confidence

        if detection is not None:
            # ── Real detection received ──
            px, py = detection["cx"], detection["cy"]
            self.last_detection_pos = (px, py)
            self.lost_frames = 0   # RESET lost_frames counter on real detection

            # Jump rejection (too far from last position)
            if self.last_position is not None:
                dist = float(np.hypot(px - self.last_position[0],
                                      py - self.last_position[1]))
                if dist > self._max_dist_jump and confidence < 0.6:
                    # Suspected false positive — skip this frame
                    self.frames_missing += 1
                    if self.initialized:
                        self._ekf_predict()
                        self._store_predicted_position()
                    self._check_and_reset_if_diverged()
                    return self._check_status()

            # Pixel-to-world coordinate conversion
            try:
                wx, wy = self.p2w.to_world(px, py)
            except Exception:
                # Degenerate homography projection
                self.frames_missing += 1
                if self.initialized:
                    self._ekf_predict()
                    self._store_predicted_position()
                return self._check_status()

            # Check world coords are physically plausible
            if abs(wx) > 100 or abs(wy) > 50:
                # Degenerate projection — ignore
                self.frames_missing += 1
                if self.initialized:
                    self._ekf_predict()
                    self._store_predicted_position()
                return self._check_status()

            # ── EKF initialization or update ──
            if not self.initialized:
                # First detection — initialize state
                self._init_state(wx, wy)

            elif len(self._world_positions) == 1:
                # Second detection — seed velocity estimate
                prev_wx, prev_wy = self._world_positions[0]
                vx_s = float(np.clip((wx - prev_wx) / DT, -30.0, 30.0))
                vy_s = float(np.clip((wy - prev_wy) / DT, -30.0, 30.0))
                self.x[2, 0] = vx_s
                self.x[3, 0] = vy_s
                # Predict then update
                self._ekf_predict()
                self._ekf_update(np.array([[wx], [wy]]))

            else:
                # Steady-state: adaptive measurement noise based on confidence
                r_scale = 0.5 if confidence > 0.7 else (1.0 if confidence > 0.4 else 3.0)
                self.R = np.eye(2) * R_POS * r_scale
                self._ekf_predict()
                self._ekf_update(np.array([[wx], [wy]]))

            # ── Store world position history ──
            self._world_positions.append((wx, wy))

            # ── Convert EKF state back to pixel space for display ──
            ex_px, ex_py = self.p2w.to_pixel(self.x[0, 0], self.x[1, 0])
            self.raw_positions.append((px, py))
            self.positions.append((ex_px, ex_py))
            self.last_position = (ex_px, ex_py)

            # ── Update velocity estimate ──
            if len(self.positions) >= 2:
                p1 = list(self.positions)[-2]
                p2 = list(self.positions)[-1]
                self.velocity = (float(p2[0] - p1[0]), float(p2[1] - p1[1]))

            self.predicted_position = (ex_px, ex_py)
            self.frames_missing = 0
            self.is_tracking = True
            self.consecutive_good += 1

        else:
            # ── No detection: coasting or reset ──
            self.frames_missing += 1
            self.lost_frames += 1  # Increment lost_frames counter
            self.consecutive_good = 0
            self.last_detection_pos = None

            # HARD RESET if lost too long
            if self.lost_frames > COAST_FRAMES:
                # More than 10 frames without a detection → complete reset
                self.reset()
                return False

            # PREDICT-ONLY for lost_frames in [1, 8]
            if self.initialized and self.frames_missing <= self._max_frames_missing:
                self._ekf_predict()
                self._store_predicted_position()

                # Clamp predicted position to frame bounds
                if self.predicted_position is not None:
                    ppx, ppy = self.predicted_position
                    if not (0 <= ppx <= _FRAME_W and 0 <= ppy <= _FRAME_H):
                        # Predicted position left frame → reset
                        self.reset()
                        return False

        # ── Final stability checks ──
        self._check_and_reset_if_diverged()

        return self._check_status()

    def _store_predicted_position(self):
        """
        Convert current EKF state to pixel space and store.

        Used during predict-only (coast) phases to maintain position
        history even without new detections.
        """
        try:
            px, py = self.p2w.to_pixel(self.x[0, 0], self.x[1, 0])
        except Exception:
            return

        self.positions.append((px, py))
        self.last_position = (px, py)
        self.predicted_position = (px, py)

        if len(self.positions) >= 2:
            p1 = list(self.positions)[-2]
            self.velocity = (float(px - p1[0]), float(py - p1[1]))

    def _check_status(self):
        """
        Update is_tracking flag based on frames_missing.

        Returns:
            bool: True if tracking is active
        """
        if self.frames_missing > self._max_frames_missing:
            self.is_tracking = False
        return self.is_tracking

    # ──────────────────────────────────────────────────────────────────
    # Forward Projection (for trajectory prediction)
    # ──────────────────────────────────────────────────────────────────

    def predict_trajectory(self, n_steps=None):
        """
        EKF forward projection: propagate into the future.

        Only runs if:
          - EKF is initialized
          - Enough observations (>= 2)
          - Covariance is healthy (trace(P) <= 500)

        Args:
            n_steps: number of steps to project (default: FORWARD_STEPS)

        Returns:
            list of (px, py) pixel tuples along predicted arc
        """
        if not self.initialized or len(self._world_positions) < 2:
            return []

        if not self.covariance_healthy():
            return []

        n_steps = n_steps if n_steps is not None else FORWARD_STEPS

        # Save current state (don't modify during projection)
        x_save = self.x.copy()
        P_save = self.P.copy()

        self._future_positions = []
        self._future_covs = []

        x_fwd = self.x.copy()
        P_fwd = self.P.copy()

        for _ in range(n_steps):
            # Predict step
            F = self._jacobian_F(x_fwd)
            x_fwd = self._f(x_fwd)
            P_fwd_new = F @ P_fwd @ F.T + self.Q
            P_fwd_new = 0.5 * (P_fwd_new + P_fwd_new.T)

            if not np.isfinite(P_fwd_new).all():
                P_fwd_new = P_fwd.copy()

            P_fwd = P_fwd_new

            # Check state validity
            if not np.isfinite(x_fwd).all():
                break

            # Convert to pixel space
            try:
                px, py = self.p2w.to_pixel(float(x_fwd[0, 0]), float(x_fwd[1, 0]))
            except Exception:
                break

            self._future_positions.append((px, py))
            self._future_covs.append(P_fwd[:2, :2].copy())

        # Restore original state
        self.x = x_save
        self.P = P_save

        return list(self._future_positions)

    def get_uncertainty_ellipse(self, step):
        """
        Return 66% confidence ellipse for a future projection step.

        Only returns valid ellipse if:
          - covariance_healthy() is True (trace(P) <= 200 AND lost_frames == 0)
          - step is within the projected trajectory

        Args:
            step: step index into the predicted trajectory

        Returns:
            dict with keys {center, axes, angle} or None
        """
        # Don't draw ellipses if covariance is unhealthy
        # (trace(P) > 200 or actively coasting)
        if not self.covariance_healthy():
            return None

        if step >= len(self._future_positions) or step >= len(self._future_covs):
            return None

        center = self._future_positions[step]
        P2 = self._future_covs[step]

        if not np.isfinite(P2).all():
            return None

        P2 = 0.5 * (P2 + P2.T)

        # Eigenvalue decomposition for ellipse axes
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(P2)
        except np.linalg.LinAlgError:
            return None

        if not np.isfinite(eigenvalues).all():
            return None

        # Scale eigenvalues to pixels
        pixel_scale = self._estimate_pixel_scale()
        a_px = max(4, int(CHI2_66_2DOF * np.sqrt(max(eigenvalues[1], 1e-9)) * pixel_scale))
        b_px = max(2, int(CHI2_66_2DOF * np.sqrt(max(eigenvalues[0], 1e-9)) * pixel_scale))
        angle = float(np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])))

        return {"center": center, "axes": (a_px, b_px), "angle": angle}

    def _estimate_pixel_scale(self):
        """
        Estimate the world-to-pixel scale factor at current position.

        Used to convert world-space covariance to pixel space for visualization.

        Returns:
            float: pixels per metre
        """
        if not self.initialized:
            return 100.0

        try:
            wx, wy = float(self.x[0, 0]), float(self.x[1, 0])
            px0, py0 = self.p2w.to_pixel(wx, wy)
            px1, py1 = self.p2w.to_pixel(wx + 1.0, wy)
            return float(np.hypot(px1 - px0, py1 - py0))
        except Exception:
            return 100.0

    # ──────────────────────────────────────────────────────────────────
    # State Telemetry
    # ──────────────────────────────────────────────────────────────────

    def get_ekf_state(self):
        """
        Return current EKF state for telemetry display.

        Returns:
            dict with keys:
              px_m, py_m: position (metres)
              vx_ms, vy_ms: velocity (m/s)
              beta: drag factor
              speed_ms: scalar speed (m/s)
              P_pos: 2x2 position covariance (metres²)
        """
        if not self.initialized:
            return {
                "px_m": 0.0,
                "py_m": 0.0,
                "vx_ms": 0.0,
                "vy_ms": 0.0,
                "beta": BETA_INIT,
                "speed_ms": 0.0,
                "P_pos": np.eye(2),
            }

        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])

        return {
            "px_m": float(self.x[0, 0]),
            "py_m": float(self.x[1, 0]),
            "vx_ms": vx,
            "vy_ms": vy,
            "beta": float(self.x[4, 0]),
            "speed_ms": float(np.hypot(vx, vy)),
            "P_pos": self.P[:2, :2].copy(),
        }

    # ──────────────────────────────────────────────────────────────────
    # Public Interface (drop-in compatibility)
    # ──────────────────────────────────────────────────────────────────

    def get_positions(self):
        """Return tracked positions in pixel space."""
        return list(self.positions)

    def get_raw_positions(self):
        """Return unsmoothed raw detection positions."""
        return list(self.raw_positions)

    def get_frames_missing(self):
        """Return number of consecutive frames without detection."""
        return self.frames_missing

    def get_velocity(self):
        """Return estimated velocity (dx, dy) pixels/frame."""
        return self.velocity

    def get_speed(self):
        """Return scalar speed in pixels/frame."""
        return float(np.hypot(*self.velocity))

    def has_enough_data(self):
        """Return True if enough positions for prediction."""
        return len(self.positions) >= self._min_pts

    def get_predicted_position(self):
        """Return current EKF predicted pixel position."""
        return self.predicted_position

    def get_last_detection_position(self):
        """Return last raw detection pixel position."""
        return self.last_detection_pos

    def get_confidence(self):
        """Return current detection confidence."""
        return self.current_confidence

    def is_using_prediction(self):
        """Return True if in predict-only mode (lost_frames > 0)."""
        return self.lost_frames > 0 and self.is_tracking

    def get_smoothed_positions(self):
        """Return moving-average smoothed positions."""
        import utils
        return utils.smooth_positions(list(self.positions))

    def reset(self):
        """
        Full hard reset of all tracking state.

        Clears:
          - EKF state and covariance
          - Position history and trajectory
          - All frame counters
          - Velocity and confidence estimates
        """
        self.x = np.zeros((5, 1), dtype=np.float64)
        self.P = np.eye(5, dtype=np.float64)
        self.initialized = False

        self.positions.clear()
        self.raw_positions.clear()
        self._world_positions.clear()
        self._future_positions.clear()
        self._future_covs.clear()

        self.frames_missing = 0
        self.lost_frames = 0
        self.is_tracking = False
        self.last_position = None
        self.last_detection_pos = None
        self.predicted_position = None
        self.velocity = (0.0, 0.0)
        self.current_confidence = 0.0
        self.consecutive_good = 0
