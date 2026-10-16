"""A small frame-level boundary between vision algorithms and live display.

Future trackers can consume observations; another detector can produce the
same result without changing camera capture or the preview windows.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .processing import Box, process_frame
from .recognition import DigitRecognizer


@dataclass(frozen=True)
class Observation:
    """One detected subject in camera pixel coordinates."""

    box: Box
    center: tuple[int, int]
    label: str | None
    score: float | None
    polygon: np.ndarray | None = None
    source_score: float | None = None  # Paper detector score, if two-stage.
    is_unknown: bool = False  # A literal ASCII '?' can be a valid OCR result.


@dataclass
class DetectionResult:
    observations: list[Observation]
    gray: np.ndarray | None = None
    binary: np.ndarray | None = None
    regions: list[np.ndarray] = field(default_factory=list)
    region_name: str = "Regions"
    score_name: str = "score"
    diagnostics: dict = field(default_factory=dict)
    debug_crops: dict[str, np.ndarray] = field(default_factory=dict)


class FrameDetector(Protocol):
    """A detector consumes one BGR frame and returns independent observations."""

    def detect(self, frame: np.ndarray) -> DetectionResult: ...


class FrameSource(Protocol):
    """Camera-like input; a hardware camera adapter only needs these methods."""

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class CardRegionReader(FrameDetector, Protocol):
    def detect_in_boxes(
        self, frame: np.ndarray, paper_boxes: list[Box] | None
    ) -> DetectionResult: ...


class NumberCardDetector:
    """Current stage's paper and digit implementation of FrameDetector."""

    def __init__(
        self, mode: str = "adaptive", min_area: int = 120, recognize: bool = True
    ) -> None:
        self.mode = mode
        self.min_area = min_area
        self.recognizer = DigitRecognizer() if recognize else None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return self.detect_in_boxes(frame, None)

    def detect_in_boxes(
        self, frame: np.ndarray, paper_boxes: list[Box] | None
    ) -> DetectionResult:
        """Read digits only inside externally supplied paper detections."""
        analysis = process_frame(
            frame, self.mode, self.min_area, paper_boxes=paper_boxes
        )
        observations: list[Observation] = []
        for candidate in analysis.candidates:
            x, y, width, height = candidate.box
            if self.recognizer is None:
                label, score = None, None
            else:
                label, score = self.recognizer.predict(candidate.mask)
            observations.append(
                Observation(
                    candidate.box,
                    (x + width // 2, y + height // 2)
                    if candidate.polygon is None else
                    tuple(int(round(value)) for value in candidate.polygon.mean(axis=0)),
                    label,
                    score,
                    candidate.polygon,
                    is_unknown=label == "?",
                )
            )
        return DetectionResult(
            observations, analysis.gray, analysis.binary, analysis.papers,
            region_name="Papers", score_name="sim",
        )
