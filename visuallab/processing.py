"""Locate bright paper surfaces before looking for dark digit strokes."""

from dataclasses import dataclass
import cv2
import numpy as np

from .geometry import paper_quad, rectify_paper

Box = tuple[int, int, int, int]  # x, y, width, height


@dataclass
class GlyphCandidate:
    box: Box
    mask: np.ndarray  # A cropped stroke, rectified when four corners are reliable.
    polygon: np.ndarray | None = None  # Stroke box projected into the camera frame.


@dataclass
class FrameAnalysis:
    gray: np.ndarray
    binary: np.ndarray
    papers: list[np.ndarray]  # Paper contours; touching cards may share one region.
    candidates: list[GlyphCandidate]


def _box_overlap(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / min(aw * ah, bw * bh)


def _near_paper_boundary(box: Box, paper_box: Box, margin: int) -> bool:
    x, y, width, height = box
    px, py, paper_width, paper_height = paper_box
    return (
        x - px <= margin
        or y - py <= margin
        or px + paper_width - (x + width) <= margin
        or py + paper_height - (y + height) <= margin
    )


def _split_edge_attachment(
    box: Box, glyph: np.ndarray, paper_box: Box, min_area: int
) -> tuple[Box, np.ndarray]:
    """Try to detach a thin paper edge joined to a much larger digit stroke.

    This is deliberately conditional: opening every glyph would erase thin
    handwritten strokes. A suspect component must span much of the paper and
    reach its boundary; the recovered inner part must retain substantial ink.
    """
    x, y, width, height = box
    _, _, paper_width, paper_height = paper_box
    margin = max(6, round(0.05 * min(paper_width, paper_height)))
    if not (
        (width >= 0.6 * paper_width or height >= 0.6 * paper_height)
        and _near_paper_boundary(box, paper_box, margin)
    ):
        return box, glyph

    opened = cv2.morphologyEx(
        glyph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    original_pixels = np.count_nonzero(glyph)
    options: list[tuple[int, Box, np.ndarray]] = []
    for index in range(1, count):
        sx, sy, sw, sh, pixels = map(int, stats[index])
        inner_box = (x + sx, y + sy, sw, sh)
        if (
            pixels < max(min_area, 0.3 * original_pixels)
            or _near_paper_boundary(inner_box, paper_box, margin)
            or (sw > 0.85 * width and sh > 0.85 * height)
        ):
            continue
        inner = np.zeros((sh, sw), dtype=np.uint8)
        inner[labels[sy:sy + sh, sx:sx + sw] == index] = 255
        options.append((pixels, inner_box, inner))
    if not options:
        return box, glyph
    _, clean_box, clean_glyph = max(options, key=lambda option: option[0])
    return clean_box, clean_glyph


def _paper_regions(
    frame: np.ndarray, gray: np.ndarray, max_area_fraction: float = .55,
) -> list[np.ndarray]:
    """Combine pale-surface regions with four-corner paper outlines.

    The surface route tolerates touching cards; the edge route can recover a
    white card on a similarly colored background. Neither route treats an
    arbitrary dark shape in the room as a digit.
    """
    height, width = gray.shape
    frame_area = height * width
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Strict and relaxed color proposals complement each other: a relaxed
    # threshold recovers warm paper, while the strict one keeps a white card
    # separate when the relaxed mask merges it with a bright object.
    surface_contours: list[np.ndarray] = []
    for saturation_limit in (20, 40, 65):
        pale = cv2.inRange(hsv, (0, 0, 145), (179, saturation_limit, 255))
        pale = cv2.morphologyEx(pale, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(
            pale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        surface_contours.extend(sorted(contours, key=cv2.contourArea, reverse=True))

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 12, 40)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    papers: list[np.ndarray] = []
    for contour in surface_contours:
        area = cv2.contourArea(contour)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not (
            max(1200, frame_area * 0.004) <= area <= frame_area * max_area_fraction
            and box_width >= 50
            and box_height >= 50
            and 0.35 <= box_width / box_height <= 4
            and max(
                area / (box_width * box_height),
                area / max(1, np.prod(cv2.minAreaRect(contour)[1])),
            ) >= 0.68
            and area / cv2.contourArea(cv2.convexHull(contour)) >= 0.86
        ):
            continue
        box = (x, y, box_width, box_height)
        if any(_box_overlap(box, cv2.boundingRect(existing)) > 0.6 for existing in papers):
            continue
        papers.append(contour)

    for contour in sorted(edge_contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)
        quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        area = cv2.contourArea(quad)
        x, y, box_width, box_height = cv2.boundingRect(quad)
        if not (
            max(1200, frame_area * 0.004) <= area <= frame_area * max_area_fraction
            and box_width >= 50
            and box_height >= 50
            and 0.35 <= box_width / box_height <= 4
        ):
            continue
        # A candidate rectangle must contain a pale surface, not just an edge.
        interior = np.zeros_like(gray)
        cv2.drawContours(interior, [quad], -1, 255, -1)
        inside = interior > 0
        if np.median(hsv[:, :, 2][inside]) < 145 or np.median(
            hsv[:, :, 1][inside]
        ) > 65:
            continue
        box = (x, y, box_width, box_height)
        center = (x + box_width // 2, y + box_height // 2)
        if any(
            _box_overlap(box, cv2.boundingRect(existing)) > 0.6
            or cv2.pointPolygonTest(existing, center, False) >= 0
            for existing in papers
        ):
            continue
        papers.append(quad)
    return papers


def _paper_proposals_from_boxes(
    frame: np.ndarray, gray: np.ndarray, boxes: list[Box]
) -> list[tuple[np.ndarray, bool]]:
    """Refine each detector box locally; a bare box is never a perspective quad."""
    frame_height, frame_width = gray.shape
    proposals: list[tuple[np.ndarray, bool]] = []
    for x, y, width, height in boxes:
        # The caller may receive partly off-screen detector boxes.
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_width, x + width), min(frame_height, y + height)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        box = (x1, y1, x2 - x1, y2 - y1)
        margin = round(0.22 * max(box[2], box[3]))
        rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
        rx2, ry2 = min(frame_width, x2 + margin), min(frame_height, y2 + margin)
        offset = np.array([[[rx1, ry1]]], dtype=np.int32)
        best: np.ndarray | None = None
        best_overlap = 0.0
        # Within a model proposal a card can legitimately fill most of the
        # crop. The full-frame background rejection limit does not apply.
        for local in _paper_regions(
            frame[ry1:ry2, rx1:rx2], gray[ry1:ry2, rx1:rx2], max_area_fraction=.95
        ):
            candidate = local + offset
            candidate_box = cv2.boundingRect(candidate)
            overlap = _box_overlap(box, candidate_box)
            area = cv2.contourArea(candidate)
            if (
                overlap >= 0.75
                and 0.55 * box[2] * box[3] <= area <= 1.25 * box[2] * box[3]
                and overlap > best_overlap
            ):
                best, best_overlap = candidate, overlap
        if best is not None:
            # A non-quad still supplies a useful foreground mask. It cannot
            # supply a homography, but need not fall back to background pixels.
            proposals.append((best, paper_quad(best) is not None))
        else:
            # A YOLO AABB locates a region; its corners are not paper corners.
            rectangle = np.array(
                [[[x1, y1]], [[x2 - 1, y1]], [[x2 - 1, y2 - 1]], [[x1, y2 - 1]]],
                dtype=np.int32,
            )
            proposals.append((rectangle, False))
    return proposals


def locate_papers(
    frame: np.ndarray, gray: np.ndarray | None = None,
    paper_boxes: list[Box] | None = None,
) -> list[tuple[np.ndarray, bool]]:
    """Return paper proposals independently of any glyph shape or alphabet."""
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if paper_boxes is None:
        return [(paper, True) for paper in _paper_regions(frame, gray)]
    return _paper_proposals_from_boxes(frame, gray, paper_boxes)


def process_frame(
    frame: np.ndarray, mode: str = "adaptive", min_area: int = 120,
    *, paper_boxes: list[Box] | None = None,
) -> FrameAnalysis:
    """Return dark strokes; optional detector boxes restrict paper search."""
    # OpenCV 读入的彩色帧采用 BGR 通道顺序。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if mode not in ("otsu", "adaptive"):
        raise ValueError(f"未知二值化模式: {mode}")

    # Binary projects accepted strokes back into camera coordinates. For a
    # tilted card, recognition itself uses the frontal, rectified stroke.
    binary = np.zeros_like(gray)
    proposals = locate_papers(frame, gray, paper_boxes)
    papers: list[np.ndarray] = []
    candidates: list[GlyphCandidate] = []
    for paper, can_rectify in proposals:
        paper_box = cv2.boundingRect(paper)
        px, py, paper_width, paper_height = paper_box
        # Fill the paper contour so its dark digit holes remain eligible.
        paper_mask = np.zeros((paper_height, paper_width), dtype=np.uint8)
        local_contour = paper - np.array([[[px, py]]], dtype=np.int32)
        cv2.drawContours(paper_mask, [local_contour], -1, 255, -1)
        local_gray = blurred[py:py + paper_height, px:px + paper_width]
        quad = paper_quad(paper) if can_rectify else None
        to_source = None
        if quad is not None:
            # Four supported corners allow a true perspective transform. If
            # they are hidden or merged, keep the unwarped conservative path.
            local_gray, paper_mask, to_source = rectify_paper(
                local_gray, paper_mask, quad - np.array([px, py], dtype=np.float32)
            )
        view_height, view_width = local_gray.shape
        # A proportional inset excludes torn edges at different camera distances.
        inset = max(3, round(0.035 * min(view_width, view_height)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
        paper_mask = cv2.erode(
            paper_mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        if mode == "otsu":
            pixels = local_gray[paper_mask > 0]
            if pixels.size == 0:
                continue
            # Otsu sees only this paper, not the brightness of the whole room.
            threshold, _ = cv2.threshold(
                pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            local_binary = cv2.threshold(local_gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            # The window must be wider than a thick marker stroke; 31 px made
            # the supplied handwritten digits hollow rather than solid.
            block = max(31, round(0.4 * min(view_width, view_height)) | 1)
            block = min(block, min(local_gray.shape) | 1)
            if block > min(local_gray.shape):
                block -= 2
            local_binary = cv2.adaptiveThreshold(
                local_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block, 11
            )
        ink = cv2.bitwise_and(local_binary, paper_mask)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        accepted_on_paper = False
        for index in range(1, count):
            lx, ly, width, height, pixels = map(int, stats[index])
            if pixels < min_area:
                continue
            # A connected component can accidentally contain a long paper
            # edge. Split that narrow attachment before measuring the glyph.
            glyph = np.zeros((height, width), dtype=np.uint8)
            glyph[labels[ly:ly + height, lx:lx + width] == index] = 255
            local_box, glyph = _split_edge_attachment(
                (lx, ly, width, height), glyph,
                (0, 0, view_width, view_height), min_area,
            )
            lx, ly, width, height = local_box
            pixels = np.count_nonzero(glyph)
            center_x = (lx + width / 2) / view_width
            center_y = (ly + height / 2) / view_height
            if not (
                pixels >= min_area
                and width >= max(6, view_width * 0.015)
                and height >= max(20, view_height * 0.08)
                and width <= view_width * 0.8
                and height <= view_height * 0.9
                and 0.05 <= pixels / (width * height) <= 0.78
                and 0.08 <= center_x <= 0.92
                and 0.08 <= center_y <= 0.92
            ):
                continue
            polygon = None
            if to_source is None:
                box = (px + lx, py + ly, width, height)
                projected = glyph
            else:
                # Map the four glyph-box corners and its pixels back to the
                # camera frame. The recognizer still gets the straightened mask.
                corners = np.array(
                    [[[lx, ly], [lx + width, ly],
                      [lx + width, ly + height], [lx, ly + height]]],
                    dtype=np.float32,
                )
                polygon = cv2.perspectiveTransform(corners, to_source)[0]
                polygon += np.array([px, py], dtype=np.float32)
                polygon = np.rint(polygon).astype(np.int32)
                view_stroke = np.zeros((view_height, view_width), dtype=np.uint8)
                view_stroke[ly:ly + height, lx:lx + width] = glyph
                projected = cv2.warpPerspective(
                    view_stroke, to_source, (paper_width, paper_height),
                    flags=cv2.INTER_NEAREST,
                )
                points = cv2.findNonZero(projected)
                if points is None:
                    continue
                sx, sy, sw, sh = cv2.boundingRect(points)
                box = (px + sx, py + sy, sw, sh)
            if any(_box_overlap(box, old.box) > 0.6 for old in candidates):
                continue
            candidates.append(GlyphCandidate(box, glyph, polygon))
            if to_source is None:
                x, y, _, _ = box
                binary[y:y + height, x:x + width] = cv2.bitwise_or(
                    binary[y:y + height, x:x + width], projected
                )
            else:
                binary[py:py + paper_height, px:px + paper_width] = cv2.bitwise_or(
                    binary[py:py + paper_height, px:px + paper_width], projected
                )
            accepted_on_paper = True
        # A pale cabinet or bag is not displayed as a number card unless it
        # contains an isolated, centrally placed stroke candidate.
        if accepted_on_paper:
            papers.append(paper)
    return FrameAnalysis(gray, binary, papers, candidates)
