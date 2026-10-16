"""Perspective geometry for a visible, approximately rectangular card."""

import cv2
import numpy as np


def paper_quad(contour: np.ndarray) -> np.ndarray | None:
    """Return four ordered corners only when the contour supports a quad."""
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    perimeter = cv2.arcLength(hull, True)
    for fraction in (0.01, 0.02, 0.03, 0.04):
        corners = cv2.approxPolyDP(hull, fraction * perimeter, True)
        if len(corners) != 4 or not cv2.isContourConvex(corners):
            continue
        # A clipped or partly hidden card can resemble a quad while its four
        # fitted corners are wrong. Such shapes should use the fallback path.
        if cv2.contourArea(corners) / hull_area < 0.95:
            continue
        points = corners.reshape(4, 2).astype(np.float32)
        center = points.mean(axis=0)
        # atan2 sorts a convex quad clockwise in image coordinates. Start at
        # the upper-left corner so the destination has a stable orientation.
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        points = points[np.argsort(angles)]
        points = np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)
        sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        if min(sides) < 30 or min(sides) / max(sides) < 0.2:
            continue
        return points
    return None


def rectify_paper(
    gray_roi: np.ndarray, mask_roi: np.ndarray, quad_roi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp a paper to a frontal view; inverse maps it back to the source ROI."""
    top_left, top_right, bottom_right, bottom_left = quad_roi
    width = round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    height = round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    # Large still images need not turn into huge per-card masks. A 1200-pixel
    # side preserves marker strokes while keeping a live pipeline practical.
    scale = min(1.0, 1200 / max(width, height))
    width, height = max(2, round(width * scale)), max(2, round(height * scale))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    to_view = cv2.getPerspectiveTransform(quad_roi, destination)
    to_source = cv2.getPerspectiveTransform(destination, quad_roi)
    view = cv2.warpPerspective(gray_roi, to_view, (width, height))
    paper_mask = cv2.warpPerspective(
        mask_roi, to_view, (width, height), flags=cv2.INTER_NEAREST
    )
    return view, paper_mask, to_source
