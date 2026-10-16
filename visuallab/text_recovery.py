"""Bounded geometric recovery for unreadable text, independent of characters.

No weights are trained here. A detection rectangle is never treated as four
paper corners. Gray pixels are authoritative; scores are heuristic gates,
not calibrated probabilities of the recovered text being correct.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .geometry import paper_quad
from .processing import _paper_regions
from .text_ocr import TextReader


@dataclass
class TextView:
    image: np.ndarray
    to_source: np.ndarray  # Crop pixels -> original view pixels.
    method: str
    angle: float = 0.


def rotate_view(image: np.ndarray, angle: float) -> TextView:
    """Rotate without cutting corners; retain the exact inverse for overlays."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1.)
    # cos(90°) is a tiny nonzero float: without this tolerance ceil can add
    # an extra pixel and introduce an unwanted half-pixel translation.
    rw = math.ceil(abs(matrix[0,0])*(w-1) + abs(matrix[0,1])*(h-1)-1e-9) + 1
    rh = math.ceil(abs(matrix[0,0])*(h-1) + abs(matrix[0,1])*(w-1)-1e-9) + 1
    matrix[:,2] += [(rw-w)/2, (rh-h)/2]
    forward = np.vstack([matrix, [0.,0.,1.]])
    rotated = cv2.warpAffine(image, matrix, (rw,rh), borderValue=255)
    return TextView(rotated, np.linalg.inv(forward), "rotation", float(angle))


def foreground_view(gray: np.ndarray) -> TextView | None:
    """Isolate interior dark ink without deleting dots or assuming a glyph.

    Border-connected components often belong to the hand or background in
    a tilted AABB. If no interior foreground survives, abstain. The retained
    ink keeps its original grayscale values; no binary OCR score is used.
    """
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = gray.shape
    keep = []
    for index in range(1, count):
        x,y,bw,bh,area = stats[index]
        if x <= 1 or y <= 1 or x+bw >= w-1 or y+bh >= h-1:
            continue
        if area >= 4:
            keep.append(index)
    selected = np.isin(labels, keep)
    points = cv2.findNonZero(selected.astype(np.uint8))
    if points is None or len(points) < 15:
        return None
    # The nearest axis supplies a small deskew, not an assertion that an
    # isolated character's reading direction has been established.
    rectangle = cv2.minAreaRect(points)
    edge = cv2.boxPoints(rectangle)[1] - cv2.boxPoints(rectangle)[0]
    angle = (math.degrees(math.atan2(float(edge[1]), float(edge[0])))+45) % 90-45
    isolated = np.full_like(gray, 255)
    isolated[selected] = gray[selected]
    rotated = rotate_view(isolated, angle)
    source_to_rotated = np.linalg.inv(rotated.to_source)
    locations = cv2.perspectiveTransform(points.astype(np.float32), source_to_rotated)
    x,y,bw,bh = cv2.boundingRect(locations)
    padding = max(6, round(.15*min(bw,bh)))
    rh,rw = rotated.image.shape
    x1,y1 = max(0,x-padding), max(0,y-padding)
    x2,y2 = min(rw,x+bw+padding), min(rh,y+bh+padding)
    translation = np.array([[1,0,x1],[0,1,y1],[0,0,1]], np.float64)
    return TextView(rotated.image[y1:y2,x1:x2], rotated.to_source @ translation,
                    "foreground_deskew", angle)


def visible_paper_quads(image: np.ndarray, text_quad: np.ndarray | None = None) -> list[np.ndarray]:
    """Optional pale planar carrier geometry; other carriers can fall back.

    Work on a bounded image, and reject clipped/unsupported quads. This is
    a geometry aid after a text detection, never a source of text proposals.
    """
    offset = np.zeros(2,np.float32)
    if text_quad is not None:
        x,y,w,h = cv2.boundingRect(text_quad.astype(np.float32))
        margin = round(.6*max(w,h))
        left,top = max(0,x-margin),max(0,y-margin)
        right,bottom = min(image.shape[1],x+w+margin),min(image.shape[0],y+h+margin)
        image = image[top:bottom,left:right]
        offset = np.array([left,top],np.float32)
    h,w = image.shape[:2]
    if min(h,w) < 3:
        return []
    # Recovering one word must not rescan a high-resolution room. The local
    # boundary is also rejected below, so it cannot become a fake paper edge.
    scale = min(1., (480 if text_quad is not None else 960)/max(h,w))
    small = cv2.resize(image, (max(1,round(w*scale)), max(1,round(h*scale))))
    sh,sw = small.shape[:2]
    quads = []
    for contour in _paper_regions(small, cv2.cvtColor(small,cv2.COLOR_BGR2GRAY), .95):
        quad = paper_quad(contour)
        if quad is None or (quad[:,0] <= 2).any() or (quad[:,1] <= 2).any():
            continue
        if (quad[:,0] >= sw-3).any() or (quad[:,1] >= sh-3).any():
            continue
        quads.append(quad*np.array([w/sw,h/sh],np.float32)+offset)
    return quads


def planar_text_view(image: np.ndarray, text_quad: np.ndarray,
                     carriers: list[np.ndarray]) -> TextView | None:
    """Use observed carrier corners, then crop the same text in paper space."""
    text_area = abs(cv2.contourArea(text_quad))
    options = []
    for quad in carriers:
        area = abs(cv2.contourArea(quad))
        if not 1.3*text_area <= area <= 30*text_area:
            continue
        intersection, _ = cv2.intersectConvexConvex(quad.astype(np.float32), text_quad.astype(np.float32))
        if intersection >= .9*text_area:
            options.append(quad)
    if not options:
        return None
    quad = min(options, key=cv2.contourArea)
    width = max(np.linalg.norm(quad[1]-quad[0]),np.linalg.norm(quad[2]-quad[3]))
    height = max(np.linalg.norm(quad[3]-quad[0]),np.linalg.norm(quad[2]-quad[1]))
    scale = min(1.,800/max(width,height))
    width,height = max(2,round(width*scale)),max(2,round(height*scale))
    destination = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    forward = cv2.getPerspectiveTransform(quad.astype(np.float32),destination)
    paper = cv2.warpPerspective(image,forward,(width,height),borderValue=(255,255,255))
    projected = cv2.perspectiveTransform(text_quad[None].astype(np.float32),forward)[0]
    x,y,w,h = cv2.boundingRect(projected)
    x1,y1,x2,y2 = max(0,x),max(0,y),min(width,x+w),min(height,y+h)
    if min(x2-x1,y2-y1) < 3:
        return None
    translation = np.array([[1,0,x1],[0,1,y1],[0,0,1]],np.float64)
    return TextView(cv2.cvtColor(paper[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY),
                    np.linalg.inv(forward) @ translation,"paper_homography")


def recover_text(reader: TextReader, gray: np.ndarray, *,
                 planar: TextView | None = None,
                 min_score: float = .85) -> tuple[TextView | None, str, float, dict]:
    """At most six recovery OCR calls: one plane, three directions, two checks.

    Caller only invokes this after the normal prediction was rejected. A
    competing high-score label or inconsistent neighboring rotations causes
    abstention. We do not rotate every frame until *some* label wins.
    """
    trials = []

    def read(view: TextView) -> tuple[str,float]:
        text,score = reader.predict(cv2.cvtColor(view.image,cv2.COLOR_GRAY2BGR))
        trials.append(dict(method=view.method,angle=view.angle,text=text,score=score))
        return text,score

    if planar is not None:
        text,score = read(planar)
        if text and score >= min_score:
            return planar,text,score,dict(status="recovered",trials=trials)
    foreground = foreground_view(gray)
    if foreground is None:
        return None,"",0.,dict(status="no_reliable_foreground",trials=trials)
    candidates = []
    for angle in (0.,-90.,90.):
        rotated = rotate_view(foreground.image,angle)
        view = TextView(rotated.image,foreground.to_source @ rotated.to_source,
                        "foreground_orientation",foreground.angle+angle)
        text,score = read(view)
        candidates.append((score,text,view))
    candidates.sort(key=lambda item:item[0],reverse=True)
    score,text,best = candidates[0]
    competitors = [s for s,t,_ in candidates[1:] if t and t != text]
    if not text or score < min_score or (competitors and score-max(competitors) < .15):
        return None,"",0.,dict(status="weak_or_ambiguous",trials=trials)
    # Local agreement is an extra rejection test, not a calibrated angle model.
    agreement = 1
    for offset in (-7.,7.):
        neighbor = rotate_view(best.image,offset)
        alternative,confidence = read(neighbor)
        if alternative == text and confidence >= .75:
            agreement += 1
        elif alternative and alternative != text and confidence >= .75:
            return None,"",0.,dict(status="unstable_direction",trials=trials)
    if agreement < 2:
        return None,"",0.,dict(status="unstable_direction",trials=trials)
    return best,text,score,dict(status="recovered",trials=trials)
