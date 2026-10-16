"""Local PP-OCR English recognition, independent of the user's example cards.

The ONNX vocabulary contains every printable ASCII character. Vocabulary
coverage does not guarantee recognition of every font, orientation or symbol.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import math

import cv2
import numpy as np

from .detection import DetectionResult, Observation
from .geometry import paper_quad, rectify_paper
from .processing import Box, locate_papers

DEFAULT_OCR_MODEL = Path(__file__).resolve().parents[1] / "models/en_PP-OCRv4_rec_mobile.onnx"


class TextReader(Protocol):
    def predict(self, image: np.ndarray) -> tuple[str, float]: ...


def decode_ctc(probabilities: np.ndarray, alphabet: list[str]) -> tuple[str, float]:
    """Collapse repeated tokens and blanks; preserve real punctuation and spaces."""
    indices = probabilities.argmax(axis=1)
    selected = [i for i, token in enumerate(indices)
                if token != 0 and (i == 0 or token != indices[i - 1])]
    text = "".join(alphabet[int(indices[i])] for i in selected).strip()
    score = float(np.mean([probabilities[i, indices[i]] for i in selected])) if selected else 0.0
    return text, score


class PPOcrTextReader:
    """Pretrained CTC line reader; no 1/2 templates or digit-only whitelist."""

    def __init__(self, model_path: str | Path = DEFAULT_OCR_MODEL) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"OCR 权重不存在：{path}。请先运行 python tools/setup_ocr.py")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("请先安装本地 OCR 依赖：uv sync --extra ocr") from error
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if "character" not in metadata:
            raise ValueError("OCR 模型缺少 character 元数据；请使用文档指定的 PP-OCRv4 英文模型")
        # Official RapidOCR metadata uses dictionary + space, plus CTC blank.
        self.alphabet = [""] + metadata["character"].splitlines() + [" "]
        if set(map(chr, range(32, 127))) - set(self.alphabet):
            raise ValueError("OCR 模型字符表未覆盖全部可打印 ASCII")
        if self.session.get_outputs()[0].shape[-1] != len(self.alphabet):
            raise ValueError("OCR 模型输出维度与字符表不匹配")
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        if image.ndim != 3 or min(image.shape[:2]) == 0:
            raise ValueError("OCR 输入应为非空 BGR 图像")
        height, width = image.shape[:2]
        resized_width = max(1, math.ceil(48 * width / height))
        if resized_width > 2048:
            raise ValueError("单行文本过长，请先分行；OCR 不会静默截断内容")
        tensor_width = max(320, resized_width)
        resized = cv2.resize(image, (resized_width, 48)).astype(np.float32) / 127.5 - 1
        tensor = np.zeros((1, 3, 48, tensor_width), dtype=np.float32)
        tensor[0, :, :, :resized_width] = resized.transpose(2, 0, 1)
        output = self.session.run(None, {self.input_name: tensor})[0][0]
        return decode_ctc(output, self.alphabet)


@dataclass
class PaperTextView:
    image: np.ndarray  # Original appearance retained for alternative readers.
    ink: np.ndarray    # Debug mask; not subjected to digit shape filters.
    to_frame: np.ndarray
    contour: np.ndarray


def paper_text_views(
    frame: np.ndarray, paper_boxes: list[Box] | None = None,
) -> list[PaperTextView]:
    """Rectify paper regions before reading; preserve disconnected punctuation."""
    views = []
    for contour, can_rectify in locate_papers(frame, paper_boxes=paper_boxes):
        x, y, width, height = cv2.boundingRect(contour)
        image = frame[y:y + height, x:x + width].copy()
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour - np.array([[[x, y]]], dtype=np.int32)], -1, 255, -1)
        inverse = np.eye(3)
        quad = paper_quad(contour) if can_rectify else None
        if quad is not None:
            image, mask, inverse = rectify_paper(image, mask, quad - np.array([x, y], dtype=np.float32))
        translation = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float)
        inset = max(2, round(.035 * min(mask.shape)))
        mask = cv2.erode(mask, np.ones((2 * inset + 1, 2 * inset + 1), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        values = gray[mask > 0]
        if values.size == 0:
            continue
        threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Ignore low-contrast paper texture without imposing a digit shape.
        threshold = min(threshold, float(np.median(values)) - 30)
        ink = ((gray <= threshold) & (mask > 0)).astype(np.uint8) * 255
        image[mask == 0] = 255
        views.append(PaperTextView(image, ink, translation @ inverse, contour))
    return views


class CardTextDetector:
    """Paper localization plus general single-character/line OCR."""

    def __init__(
        self, model_path: str | Path = DEFAULT_OCR_MODEL, *,
        reader: TextReader | None = None, recognize: bool = True, min_score: float = .6,
    ) -> None:
        if not 0 <= min_score <= 1:
            raise ValueError("OCR min_score 应在 0 与 1 之间")
        self.reader = (reader if reader is not None else PPOcrTextReader(model_path)) if recognize else None
        self.min_score = min_score

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return self.detect_in_boxes(frame, None)

    def detect_in_boxes(self, frame: np.ndarray, paper_boxes: list[Box] | None) -> DetectionResult:
        views = paper_text_views(frame, paper_boxes)
        binary = np.zeros(frame.shape[:2], dtype=np.uint8)
        observations = []
        for view in views:
            projected = cv2.warpPerspective(view.ink, view.to_frame,
                                            (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST)
            binary = cv2.bitwise_or(binary, projected)
            points = cv2.findNonZero(view.ink)
            if points is None:
                continue
            x, y, width, height = cv2.boundingRect(points)
            padding = max(6, round(height * .12))
            # A uniform high-contrast view removes pen hue/lighting differences.
            # All ink components remain, including punctuation and dotted letters.
            text_image = cv2.cvtColor(255 - view.ink, cv2.COLOR_GRAY2BGR)
            crop = cv2.copyMakeBorder(text_image[y:y+height, x:x+width],
                                      padding, padding, padding, padding,
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text, score = self.reader.predict(crop) if self.reader is not None else (None, None)
            unknown = self.reader is not None and (not text or score < self.min_score)
            corners = np.array([[[x, y], [x + width, y],
                                 [x + width, y + height], [x, y + height]]], dtype=np.float32)
            polygon = np.rint(cv2.perspectiveTransform(corners, view.to_frame)[0]).astype(np.int32)
            polygon[:, 0] = polygon[:, 0].clip(0, frame.shape[1] - 1)
            polygon[:, 1] = polygon[:, 1].clip(0, frame.shape[0] - 1)
            box = cv2.boundingRect(polygon)
            observations.append(Observation(box, tuple(int(v) for v in np.rint(polygon.mean(axis=0))),
                                "?" if unknown else text, score, polygon, is_unknown=unknown))
        return DetectionResult(observations, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               binary, [view.contour for view in views],
                               region_name="Papers", score_name="ocr")
