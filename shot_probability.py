"""
Monte Carlo Shot Probability Estimator for ShotIQ.

Implements the reference MATLAB approach:
  1. Fit a Gaussian to the velocity distribution: vx ~ N(μ_vx, σ_vx)
  2. Run N=500 independent EKF forward projections, each sampling a
     different vx from the Gaussian.
  3. For each simulation, propagate until the trajectory crosses
     the basket height (3.05 m above ground).
  4. Check whether the horizontal position at that crossing falls
     within the basket window.
  5. Pr = fraction of simulations that score.

Dependencies: numpy only (no scipy).
"""

import numpy as np
from ekf_tracker import (
    EKFBallTracker,
    GRAVITY, ALPHA_DRAG, DT, FORWARD_STEPS,
    BASKET_HEIGHT_M, BASKET_X_NEAR, BASKET_X_FAR, RING_TOLERANCE_M,
    BETA_INIT,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_SIMULATIONS       = 500       # Monte Carlo sample count
MAX_SIM_STEPS       = 400       # Safety cap on forward simulation steps
VX_NOISE_FLOOR      = 0.05      # Minimum σ_vx (m/s) to prevent degenerate dist.
VY_NOISE_FLOOR      = 0.05      # Minimum σ_vy (m/s)
MIN_LAUNCH_SPEED    = 0.1       # m/s — skip MC if barely moving
MIN_OBSERVATIONS    = 8         # Need at least this many world obs for MC


def _propagate_one(x0, P_state, beta):
    """
    Propagate a single EKF sample forward until it crosses BASKET_HEIGHT_M
    or until MAX_SIM_STEPS are exceeded.

    Uses Euler integration (same as EKF predict step) for speed.

    Args:
        x0: (5,) initial state [px, py, vx, vy, beta]
        P_state: not used here (noise already folded into x0 sampling)
        beta: drag factor to use for this sample

    Returns:
        (x_cross, crosses) where:
          x_cross — world x at basket-height crossing (m), or None
          crosses — True if trajectory crossed BASKET_HEIGHT_M going upward
    """
    px, py, vx, vy = float(x0[0]), float(x0[1]), float(x0[2]), float(x0[3])
    prev_py = py

    for _ in range(MAX_SIM_STEPS):
        speed = np.sqrt(vx ** 2 + vy ** 2) + 1e-9
        drag_x = ALPHA_DRAG * beta * speed * vx
        drag_y = ALPHA_DRAG * beta * speed * vy

        px = px + vx * DT
        py = py + vy * DT
        vx = vx + drag_x * DT
        vy = vy + (drag_y + GRAVITY) * DT

        # Check for crossing BASKET_HEIGHT_M (trajectory ascending past rim)
        if prev_py < BASKET_HEIGHT_M <= py or prev_py > BASKET_HEIGHT_M >= py:
            # Linear interpolation to find px at exact crossing height
            alpha_interp = (BASKET_HEIGHT_M - prev_py) / (py - prev_py + 1e-12)
            # Approximate x at crossing (we already updated px, back-interpolate)
            # Use vx at current step (already updated, close enough for Euler)
            x_cross = px - vx * DT * (1.0 - alpha_interp)
            return x_cross, True

        prev_py = py

    return None, False


class MonteCarloProbability:
    """
    Estimates shot scoring probability via Monte Carlo simulation.

    Usage:
        mc = MonteCarloProbability()
        result = mc.estimate(ekf_tracker)
        print(result["Pr"], result["mean_vx"], result["std_vx"])
    """

    def __init__(self, n_simulations=N_SIMULATIONS):
        self.n = n_simulations
        self.last_Pr    = 0.0
        self.last_mean_vx = 0.0
        self.last_std_vx  = 0.0
        self.last_mean_vy = 0.0
        self.last_std_vy  = 0.0

    def estimate(self, ekf_tracker):
        """
        Run Monte Carlo estimation using the current EKF state.

        Args:
            ekf_tracker: an EKFBallTracker instance with live state.

        Returns:
            dict with keys:
              Pr       — scoring probability (float 0–1)
              mean_vx  — mean horizontal velocity (m/s)
              std_vx   — std-dev of horizontal velocity (m/s)
              mean_vy  — mean vertical velocity (m/s)
              std_vy   — std-dev of vertical velocity (m/s)
              valid    — bool: True if MC was actually run
        """
        if not ekf_tracker.initialized:
            return self._empty_result()

        world_obs = ekf_tracker._world_positions
        if len(world_obs) < MIN_OBSERVATIONS:
            return self._empty_result()

        # ── Estimate velocity distribution from EKF state ──
        # Primary source: EKF posterior mean and covariance
        state = ekf_tracker.x.flatten()
        P     = ekf_tracker.P

        mu_vx  = float(state[2])
        mu_vy  = float(state[3])
        mu_beta = float(state[4])

        # σ_vx, σ_vy from diagonal of P (velocity submatrix)
        sigma_vx = float(np.sqrt(max(P[2, 2], VX_NOISE_FLOOR ** 2)))
        sigma_vy = float(np.sqrt(max(P[3, 3], VY_NOISE_FLOOR ** 2)))

        # Enforce noise floor
        sigma_vx = max(sigma_vx, VX_NOISE_FLOOR)
        sigma_vy = max(sigma_vy, VY_NOISE_FLOOR)

        # Skip MC if ball is barely moving
        speed = np.sqrt(mu_vx ** 2 + mu_vy ** 2)
        if speed < MIN_LAUNCH_SPEED:
            return self._empty_result()

        # Current position for sim initial conditions
        mu_px = float(state[0])
        mu_py = float(state[1])

        # ── Monte Carlo loop ──
        scores = 0
        rng = np.random.default_rng()   # thread-safe, reproducible enough

        # Sample vx ~ N(μ_vx, σ_vx) — vectorised draw
        sampled_vx = rng.normal(mu_vx, sigma_vx, self.n)
        sampled_vy = rng.normal(mu_vy, sigma_vy, self.n)
        sampled_beta = np.clip(
            rng.normal(mu_beta, max(float(np.sqrt(max(P[4, 4], 1e-6))), 0.01), self.n),
            0.0, 10.0
        )

        for i in range(self.n):
            x0 = np.array([mu_px, mu_py, sampled_vx[i], sampled_vy[i], sampled_beta[i]])
            x_cross, crossed = _propagate_one(x0, P, sampled_beta[i])

            if crossed and x_cross is not None:
                # Check if x_cross is within basket window (with ring tolerance)
                lo = BASKET_X_NEAR - RING_TOLERANCE_M
                hi = BASKET_X_FAR  + RING_TOLERANCE_M
                if lo <= x_cross <= hi:
                    scores += 1

        Pr = scores / self.n

        # Store for telemetry
        self.last_Pr      = Pr
        self.last_mean_vx = mu_vx
        self.last_std_vx  = sigma_vx
        self.last_mean_vy = mu_vy
        self.last_std_vy  = sigma_vy

        return {
            "Pr":      Pr,
            "mean_vx": mu_vx,
            "std_vx":  sigma_vx,
            "mean_vy": mu_vy,
            "std_vy":  sigma_vy,
            "valid":   True,
        }

    def _empty_result(self):
        return {
            "Pr":      0.0,
            "mean_vx": 0.0,
            "std_vx":  0.0,
            "mean_vy": 0.0,
            "std_vy":  0.0,
            "valid":   False,
        }
