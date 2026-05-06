"""
Pixel-to-World Coordinate Transformation for ShotIQ.

Uses a camera homography (perspective transform) to convert pixel
detections into real-world metres, and to project world coordinates
back to pixel space for drawing.

The default homography is computed from a standard free-throw camera
angle (camera ~3 m high, looking down at ~25 degrees toward the hoop).
Replace HOMOGRAPHY_MATRIX with one from cv2.findHomography once you
have four real calibration point pairs.

Coordinate system (world / real-world metres):
  Origin  -- beneath the free-throw line
  +x      -- toward the basket (downrange)
  +y      -- upward
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# CONFIG — replace with court-specific values
# ─────────────────────────────────────────────
# Default assumes a 1280x720 frame where:
#   Bottom-centre   --> free-throw origin (0, 0) m
#   Upper-centre    --> basket (4.57, 3.05) m
#   Left/right mid  --> lane markers at ~1.14 m wide, 0.9 m up
#
# These are approximate for a room-trial setup.
# Override by passing your own pix_pts / world_pts to __init__.

_DEFAULT_PIX_POINTS = np.array([
    [640,  700],   # free-throw origin (bottom centre)
    [320,  540],   # left lane mid
    [960,  540],   # right lane mid
    [640,  240],   # basket location (upper centre)
], dtype=np.float32)

_DEFAULT_WORLD_POINTS = np.array([
    [0.00,  0.00],
    [-1.14, 0.90],
    [ 1.14, 0.90],
    [ 4.57, 3.05],
], dtype=np.float32)


def _compute_homography(src_pts, dst_pts):
    """Compute H (src->dst) using DLT; returns 3x3 float64 matrix."""
    H, _ = cv2.findHomography(
        src_pts.astype(np.float32),
        dst_pts.astype(np.float32),
        method=0,
    )
    return H.astype(np.float64)


class PixelToWorld:
    """
    Bidirectional pixel <-> real-world coordinate transform.

    Usage:
        ptw = PixelToWorld()                  # use default homography
        wx, wy = ptw.to_world(px, py)         # pixel -> metres
        px, py = ptw.to_pixel(wx, wy)         # metres -> pixel

    For a real court, call:
        ptw = PixelToWorld(pix_pts, world_pts)
    where pix_pts / world_pts are (N>=4, 2) arrays of corresponding points.
    """

    def __init__(self, pix_pts=None, world_pts=None):
        if pix_pts is not None and world_pts is not None:
            src = np.array(pix_pts, dtype=np.float32)
            dst = np.array(world_pts, dtype=np.float32)
        else:
            src = _DEFAULT_PIX_POINTS
            dst = _DEFAULT_WORLD_POINTS

        self.H     = _compute_homography(src, dst)   # pixel -> world
        self.H_inv = _compute_homography(dst, src)   # world -> pixel

    # ------------------------------------------------------------------
    def to_world(self, px, py):
        """
        Convert one pixel coordinate to real-world metres.

        Args:
            px, py: float pixel coordinates.

        Returns:
            (wx, wy): real-world coordinates in metres.
        """
        pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
        wpt = cv2.perspectiveTransform(pt, self.H)
        return float(wpt[0, 0, 0]), float(wpt[0, 0, 1])

    def to_pixel(self, wx, wy):
        """
        Convert real-world metres back to pixel coordinates.

        Args:
            wx, wy: real-world coordinates in metres.

        Returns:
            (px, py): integer pixel coordinates.
        """
        pt = np.array([[[float(wx), float(wy)]]], dtype=np.float64)
        ppt = cv2.perspectiveTransform(pt, self.H_inv)
        return int(round(ppt[0, 0, 0])), int(round(ppt[0, 0, 1]))

    def batch_to_world(self, pixel_points):
        """
        Convert an array of pixel points to world coordinates.

        Args:
            pixel_points: list/array of (px, py) pairs.

        Returns:
            numpy array of shape (N, 2) in metres.
        """
        pts = np.array(pixel_points, dtype=np.float64).reshape(-1, 1, 2)
        wpts = cv2.perspectiveTransform(pts, self.H)
        return wpts.reshape(-1, 2)

    def batch_to_pixel(self, world_points):
        """
        Convert an array of world points to pixel coordinates.

        Args:
            world_points: list/array of (wx, wy) pairs in metres.

        Returns:
            numpy array of shape (N, 2) as integer pixel coordinates.
        """
        pts = np.array(world_points, dtype=np.float64).reshape(-1, 1, 2)
        ppts = cv2.perspectiveTransform(pts, self.H_inv)
        return ppts.reshape(-1, 2).astype(int)
