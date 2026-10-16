"""Optional local YOLO adapter. No model is loaded on the traditional path."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from time import perf_counter

import numpy as np
import cv2

from .detection import CardRegionReader, DetectionResult, NumberCardDetector, Observation


class YoloObjectDetector:
    """Map a YOLO model's class boxes into the same frame interface."""

    def __init__(
        self, weights: str | Path, *, device: str = "cpu", confidence: float = 0.25,
        image_size: int = 640, model: Any | None = None,
    ) -> None:
        if not 0 < confidence < 1:
            raise ValueError("YOLO confidence 必须在 0 与 1 之间")
        if image_size <= 0:
            raise ValueError("YOLO 输入尺寸必须大于 0")
        if model is None:
            path = Path(weights)
            if not path.is_file():
                raise FileNotFoundError(f"找不到本地 YOLO 权重：{path}")
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError("YOLO 依赖未安装：请运行 uv sync --extra yolo") from error
            model = YOLO(str(path))
        self.model = model
        self.device = device
        self.confidence = confidence
        self.image_size = image_size
        names = model.names
        self.class_names = (
            {int(index): str(name) for index, name in names.items()}
            if isinstance(names, dict) else
            {index: str(name) for index, name in enumerate(names)}
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        predictions = self.model.predict(
            source=frame, conf=self.confidence, imgsz=self.image_size,
            device=self.device, verbose=False,
        )
        height, width = frame.shape[:2]
        observations: list[Observation] = []
        for prediction in predictions:
            if prediction.boxes is None:
                continue
            # Ultralytics stores these as Nx4, N, N tensors in source pixels.
            for xyxy, confidence, class_id in zip(
                prediction.boxes.xyxy.tolist(),
                prediction.boxes.conf.tolist(),
                prediction.boxes.cls.tolist(),
            ):
                x1 = max(0, min(width, int(np.floor(xyxy[0]))))
                y1 = max(0, min(height, int(np.floor(xyxy[1]))))
                x2 = max(0, min(width, int(np.ceil(xyxy[2]))))
                y2 = max(0, min(height, int(np.ceil(xyxy[3]))))
                if x2 <= x1 or y2 <= y1:
                    continue
                label = self.class_names.get(int(class_id), f"class_{int(class_id)}")
                observations.append(
                    Observation((x1, y1, x2 - x1, y2 - y1),
                                ((x1 + x2) // 2, (y1 + y2) // 2),
                                label, float(confidence))
                )
        return DetectionResult(observations, score_name="conf")


class HybridCardDetector:
    """YOLO finds cards; existing local CV reads their contents."""

    def __init__(
        self, yolo: YoloObjectDetector, *, card_class: str = "card",
        mode: str = "adaptive", min_area: int = 120, recognize: bool = True,
        reader: CardRegionReader | None = None,
    ) -> None:
        if card_class not in yolo.class_names.values():
            examples = list(yolo.class_names.values())[:8]
            raise ValueError(
                f"权重没有 {card_class!r} 类；实际有 {len(yolo.class_names)} 类，"
                f"前几个是 {examples}。"
                "请先训练纸片检测权重，通用 COCO 权重不能直接用于 hybrid。"
            )
        self.yolo = yolo
        self.card_class = card_class
        self.reader = reader if reader is not None else NumberCardDetector(mode, min_area, recognize)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        started = perf_counter()
        found = self.yolo.detect(frame)
        object_ms = (perf_counter() - started) * 1000
        cards = [item for item in found.observations if item.label == self.card_class]
        result = self.reader.detect_in_boxes(frame, [item.box for item in cards])
        if result.diagnostics:
            result.diagnostics["timings_ms"]["object_localization"] = object_ms
            result.diagnostics["timings_ms"]["pipeline_total"] = (perf_counter()-started)*1000
        # Show detector proposals even when a card has no readable digit.
        result.regions = [
            np.array([[[x, y]], [[x + w - 1, y]],
                      [[x + w - 1, y + h - 1]], [[x, y + h - 1]]], dtype=np.int32)
            for item in cards for x, y, w, h in [item.box]
        ]
        result.region_name = "Cards"
        for index, observation in enumerate(result.observations):
            cx, cy = observation.center
            containing = [
                card.score for card in cards
                if card.box[0] <= cx < card.box[0] + card.box[2]
                and card.box[1] <= cy < card.box[1] + card.box[3]
            ]
            if containing:
                result.observations[index] = replace(
                    observation, source_score=max(containing)
                )
        return result


class YoloTextLocator:
    """Use carrier-independent text boxes through the existing OCR boundary.

    These are text AABBs, not paper corners. Paper perspective correction
    remains the responsibility of a carrier geometry provider when available.
    """

    name = "yolo_text"

    def __init__(self, detector: YoloObjectDetector, text_class: str = "text", *,
                 recover_empty: bool = False) -> None:
        if text_class not in detector.class_names.values():
            raise ValueError(f"文字定位权重必须包含 {text_class!r} 类；不能使用 card/COCO 权重代替")
        self.detector, self.text_class = detector, text_class
        self.recover_empty = recover_empty

    def locate(self, image: np.ndarray):
        from .text_regions import TextRegion
        result = self.detector.detect(image)
        # A small fine-tuned end-to-end model can emit two overlapping boxes
        # for the same glyph. Suppress duplicate text proposals before OCR;
        # neither the recognized character nor its OCR score affects this.
        items = [item for item in result.observations if item.label == self.text_class]
        to_source = np.eye(3,dtype=np.float32)
        origin = "primary"
        if not items and self.recover_empty:
            # One additional detection only after a completely empty result.
            # Rotation supplies proposals; OCR still reads original pixels,
            # so a horizontal dash is not silently relabelled as a digit 1.
            rotated = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
            retry = self.detector.detect(rotated)
            items = [item for item in retry.observations if item.label == self.text_class]
            to_source = np.array([[0,1,0],[-1,0,image.shape[0]-1],[0,0,1]],np.float32)
            origin = "rotated_detection_90"
        keep = cv2.dnn.NMSBoxes([list(item.box) for item in items],
                               [item.score for item in items], 0., .5) if items else []
        return [TextRegion(cv2.perspectiveTransform(np.array([[[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]]], np.float32),to_source)[0], item.score,origin)
                for index in np.asarray(keep).reshape(-1) for item in [items[int(index)]]
                for x,y,w,h in [item.box]]
