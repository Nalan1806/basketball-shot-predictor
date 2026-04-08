"""
Ball Detection Module for the Basketball Shot Prediction System.

Uses HSV color segmentation + contour analysis to detect an orange
basketball in each frame. Designed for robustness under varying
indoor lighting conditions.
"""

import cv2
import numpy as np
import config


class BallDetector:
    """
    Detects a basketball using color-based segmentation.

    Pipeline:
      1. Convert frame to HSV color space
      2. Create a binary mask for orange hues
      3. Apply morphological ops to clean up noise
      4. Find contours and filter by area + circularity
      5. Return the best candidate (largest valid contour)
    """

    def __init__(self):
        # HSV range for orange basketball
        self.hsv_lower = np.array(config.BALL_HSV_LOWER)
        self.hsv_upper = np.array(config.BALL_HSV_UPPER)

        # Morphological kernel for noise removal
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
        )

        # Store the last detection for debugging / visualization
        self.last_mask = None
        self.last_contours = []

    def detect(self, frame):
        """
        Detect the basketball in a single frame.

        Args:
            frame: BGR image (numpy array) from OpenCV.

        Returns:
            detection: dict with keys {x, y, w, h, cx, cy, radius, area}
                       or None if no ball found.
        """
        # Step 1: Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 2: Create color mask
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Step 3: Morphological operations to reduce noise
        # Erode removes small bright spots, dilate restores ball edges
        mask = cv2.erode(mask, self.kernel, iterations=config.MORPH_ITERATIONS)
        mask = cv2.dilate(mask, self.kernel, iterations=config.MORPH_ITERATIONS)

        # Optional: Gaussian blur for smoother contours
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        self.last_mask = mask

        # Step 4: Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.last_contours = contours

        # Step 5: Filter and select the best candidate
        best_detection = None
        best_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area bounds
            if area < config.MIN_CONTOUR_AREA or area > config.MAX_CONTOUR_AREA:
                continue

            # Filter by circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < config.MIN_CIRCULARITY:
                continue

            # This contour passes all filters — check if it's the largest
            if area > best_area:
                best_area = area

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Get minimum enclosing circle for center + radius
                (cx, cy), radius = cv2.minEnclosingCircle(contour)

                best_detection = {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "cx": int(cx),
                    "cy": int(cy),
                    "radius": int(radius),
                    "area": area,
                    "circularity": circularity
                }

        return best_detection

    def get_debug_mask(self):
        """Return the last computed mask for debug visualization."""
        return self.last_mask

    def update_hsv_range(self, lower, upper):
        """
        Dynamically update the HSV detection range.
        Useful for runtime calibration via trackbars.
        """
        self.hsv_lower = np.array(lower)
        self.hsv_upper = np.array(upper)
