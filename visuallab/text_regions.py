"""Optional text localization, independent of paper color and object classes.

DB postprocessing follows RapidOCR (Apache-2.0); see docs/TEXT_REGIONS.md.
Copyright (c) 2021 RapidOCR Authors (reference implementation).
The recognizer stays replaceable, and a selected ROI can bypass localization
for isolated characters/punctuation that a text-line detector may miss.
"""
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from .detection import DetectionResult, Observation
from .processing import Box
from .text_ocr import DEFAULT_OCR_MODEL, PPOcrTextReader, TextReader, paper_text_views

DEFAULT_TEXT_MODEL = Path(__file__).resolve().parents[1] / "models/ch_PP-OCRv4_det_mobile.onnx"


@dataclass(frozen=True)
class TextRegion:
    quad: np.ndarray  # Four text corners in the locator input's coordinates.
    score: float | None = None
    origin: str = "primary"


class TextLocator(Protocol):
    def locate(self, image: np.ndarray) -> list[TextRegion]: ...


def ordered_quad(points: np.ndarray) -> np.ndarray:
    """Order a convex text rectangle clockwise, starting near upper-left."""
    points = np.asarray(points, np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    points = points[order]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)


def crop_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Deskew a text quad; this does not infer missing paper corners."""
    points = ordered_quad(quad)
    width = max(2, round(max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3]))))
    height = max(2, round(max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1]))))
    destination = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(image, transform, (width, height),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


class DBTextLocator:
    """One pinned PP-OCRv4 mobile DB detector, CPU only, bounded input size."""

    name = "db_text"

    def __init__(self, model_path: str | Path = DEFAULT_TEXT_MODEL, *, max_side: int = 960) -> None:
        if max_side < 32:
            raise ValueError("文字定位输入最长边至少为 32")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"文字定位权重不存在：{path}；请运行 python tools/setup_text_detector.py")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("请先安装 OCR 依赖：uv sync --extra ocr") from error
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.max_side = max_side

    def locate(self, image: np.ndarray) -> list[TextRegion]:
        h, w = image.shape[:2]
        ratio = min(1., self.max_side / max(h, w))
        rw, rh = [max(32, round(value * ratio / 32) * 32) for value in (w, h)]
        # Official v3.10 preprocessing: BGR, (pixel/255 - .5)/.5, CHW.
        resized = cv2.resize(image, (rw, rh)).astype(np.float32) / 127.5 - 1
        probability = self.session.run(None, {self.input_name: resized.transpose(2, 0, 1)[None]})[0][0, 0]
        return self.regions_from_map(probability, (h, w))

    @staticmethod
    def regions_from_map(probability: np.ndarray, shape: tuple[int, int]) -> list[TextRegion]:
        """Recover DB boxes without introducing polygon/GUI dependencies."""
        mh, mw = probability.shape
        contours = cv2.findContours((probability > .3).astype(np.uint8), cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        regions = []
        for contour in contours[:1000]:
            rect = cv2.minAreaRect(contour)
            (cx, cy), (width, height), angle = rect
            if min(width, height) < 3:
                continue
            points = cv2.boxPoints(rect)
            x, y, bw, bh = cv2.boundingRect(points.astype(np.int32))
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(mw, x+bw), min(mh, y+bh)
            mask = np.zeros((y2-y1, x2-x1), np.uint8)
            cv2.fillPoly(mask, [(points - (x1, y1)).astype(np.int32)], 1)
            score = float(cv2.mean(probability[y1:y2, x1:x2], mask)[0])
            if score < .7:
                continue
            # DB expands a rectangle by area*unclip_ratio/perimeter. The
            # enclosing min-area rectangle grows each side by 2*distance;
            # this avoids Shapely/Pyclipper for rectangular output.
            distance = width * height * 2 / (2 * (width + height))
            quad = cv2.boxPoints(((cx, cy), (width+2*distance, height+2*distance), angle))
            quad *= np.array([shape[1]/mw, shape[0]/mh], np.float32)
            regions.append(TextRegion(ordered_quad(quad), score))
        return sorted(regions, key=lambda region: (float(region.quad[:, 1].mean()),
                                                   float(region.quad[:, 0].mean())))


def prepare_inputs(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exactly two variants: original gray and Otsu. Never discard punctuation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray, binary


class RegionTextDetector:
    """Object ROI -> optional text localization -> unchanged line recognizer.

    Gray is authoritative. Binary is recorded as a diagnostic; a higher
    binary score never replaces gray automatically. Scores are not calibrated
    correctness probabilities. No character-specific rules are used.
    """

    def __init__(self, *, locator: TextLocator | None = None,
                 reader: TextReader | None = None, model_path: str | Path = DEFAULT_OCR_MODEL,
                 target_roi: tuple[float, float, float, float] | None = None,
                 direct: bool = False, paper_geometry: bool = False,
                 recognize: bool = True, min_score: float = .6,
                 recovery: bool = False) -> None:
        if not 0 <= min_score <= 1:
            raise ValueError("OCR min_score 应在 0 与 1 之间")
        if target_roi is not None:
            x, y, w, h = target_roi
            if not all(np.isfinite(target_roi)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > 1 or y+h > 1:
                raise ValueError("目标 ROI 为归一化 x y w h，必须位于 0～1 内且面积大于 0")
        if not direct and locator is None:
            raise ValueError("文字定位模式需要 locator；单字符 ROI 可使用 direct")
        self.locator, self.direct, self.paper_geometry = locator, direct, paper_geometry
        self.reader = (reader if reader is not None else PPOcrTextReader(model_path)) if recognize else None
        self.target_roi, self.min_score = target_roi, min_score
        self.recovery = recovery

    def detect(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        roi = self.target_roi or (0., 0., 1., 1.)
        x1, y1 = int(roi[0]*w), int(roi[1]*h)
        x2, y2 = min(w, round((roi[0]+roi[2])*w)), min(h, round((roi[1]+roi[3])*h))
        return self.detect_in_boxes(frame, [(x1, y1, x2-x1, y2-y1)])

    def detect_in_boxes(self, frame: np.ndarray, paper_boxes: list[Box] | None) -> DetectionResult:
        started = perf_counter()
        h, w = frame.shape[:2]
        # Explicit empty proposals must never trigger a global text search.
        boxes = paper_boxes if paper_boxes is not None else [(0, 0, w, h)]
        views = []
        if self.paper_geometry and boxes:
            for view in paper_text_views(frame, boxes):
                views.append((view.image, view.to_frame, view.contour))
        else:
            for x, y, bw, bh in boxes:
                x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x+bw), min(h, y+bh)
                if x2 <= x1 or y2 <= y1:
                    continue
                contour = np.array([[x1,y1],[x2-1,y1],[x2-1,y2-1],[x1,y2-1]], np.int32)
                translation = np.array([[1,0,x1],[0,1,y1],[0,0,1]], np.float64)
                views.append((frame[y1:y2, x1:x2], translation, contour))
        geometry_ms = (perf_counter()-started)*1000
        observations, records, crops = [], [], {}
        binary_view = np.zeros((h, w), np.uint8)
        locate_ms = read_ms = recovery_ms = 0.
        recovery_count = 0  # Bound extra work to two unreadable proposals/frame.
        for view_index, (view, to_frame, contour) in enumerate(views):
            vh, vw = view.shape[:2]
            tick = perf_counter()
            regions = [TextRegion(np.array([[0,0],[vw-1,0],[vw-1,vh-1],[0,vh-1]], np.float32))] if self.direct else self.locator.locate(view)
            locate_ms += (perf_counter()-tick)*1000
            for region in regions:
                # Keep a generic margin around DB regions. Large isolated
                # glyphs may extend past the text probability map's core.
                quad = ordered_quad(region.quad.copy())
                if not self.direct:
                    quad = (quad-quad.mean(axis=0))*1.24 + quad.mean(axis=0)
                quad[:,0] = quad[:,0].clip(0, vw-1)
                quad[:,1] = quad[:,1].clip(0, vh-1)
                crop = view.copy() if self.direct else crop_quad(view, quad)
                gray, binary = prepare_inputs(crop)
                outputs = []
                for kind, image in (("gray", gray), ("binary", binary)):
                    key = f"{len(records)+1:03d}-{kind}"
                    crops[key] = image
                    tick = perf_counter()
                    text, score = self.reader.predict(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)) if self.reader else (None, None)
                    read_ms += (perf_counter()-tick)*1000
                    outputs.append({"variant":kind, "text":text, "score":score,
                                    "crop":"ocr_inputs/"+key+".png"})
                text, score = outputs[0]["text"], outputs[0]["score"]
                unknown = self.reader is not None and (not text or score < self.min_score)
                ch,cw = gray.shape
                source = np.array([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]],np.float32)
                crop_to_view = cv2.getPerspectiveTransform(source,quad.astype(np.float32))
                recovery_record = {"status":"off" if not self.recovery else "not_needed"}
                if self.recovery and unknown and recovery_count < 2:
                    from .text_recovery import recover_text, visible_paper_quads, planar_text_view
                    recovery_count += 1
                    tick = perf_counter()
                    carrier_quads = visible_paper_quads(view,quad)
                    planar = planar_text_view(view,quad,carrier_quads)
                    recovered,new_text,new_score,recovery_record = recover_text(
                        self.reader,gray,planar=planar,min_score=max(.85,self.min_score))
                    recovery_record["before"] = {"text":text,"score":score}
                    if recovered is not None:
                        # Paper maps directly to view pixels; rotation recovery
                        # maps to the initial crop. Keep this distinction exact.
                        crop_to_view = (recovered.to_source if recovered.method == "paper_homography"
                                        else crop_to_view @ recovered.to_source)
                        key = f"{len(records)+1:03d}"
                        crops[key+"-original-gray"] = gray
                        gray = recovered.image
                        binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        text,score,unknown = new_text,new_score,False
                        binary_text,binary_score = self.reader.predict(cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR))
                        outputs = [{"variant":"gray","text":text,"score":score,"crop":"ocr_inputs/"+key+"-gray.png"},
                                   {"variant":"binary","text":binary_text,"score":binary_score,"crop":"ocr_inputs/"+key+"-binary.png"}]
                        crops[key+"-gray"],crops[key+"-binary"] = gray,binary
                        recovery_record.update(method=recovered.method,angle=recovered.angle)
                    recovery_ms += (perf_counter()-tick)*1000
                elif self.recovery and unknown:
                    recovery_record = {"status":"frame_budget_exhausted"}
                polygon = cv2.perspectiveTransform(quad[None].astype(np.float32), to_frame)[0]
                polygon[:,0], polygon[:,1] = polygon[:,0].clip(0,w-1), polygon[:,1].clip(0,h-1)
                polygon = np.rint(polygon).astype(np.int32)
                observations.append(Observation(cv2.boundingRect(polygon),
                    tuple(int(v) for v in np.rint(polygon.mean(axis=0))),
                    "?" if unknown else text, score, polygon, is_unknown=unknown))
                records.append({"object_index":view_index, "text_detection_score":region.score,
                    "proposal_origin":region.origin,
                    "polygon":polygon.tolist(), "variants":outputs,
                    "disagreement":outputs[0]["text"] != outputs[1]["text"],
                    "status":"unknown" if unknown else ("localized" if self.reader is None else "accepted"),
                    "recovery":recovery_record,
                    "policy":"gray_only; binary_diagnostic"})
                # Project the *actual text crop's* binary mask, not the whole
                # object. Coordinates use the same quad as the result box.
                ch, cw = binary.shape
                if min(ch, cw) > 1:
                    # Use the floating-point transform chain, including crop
                    # rotation, rather than squeezing a rotated mask into AABB.
                    matrix = to_frame @ crop_to_view
                    projected = cv2.warpPerspective(255-binary, matrix, (w,h), flags=cv2.INTER_NEAREST)
                    binary_view = cv2.bitwise_or(binary_view, projected)
        return DetectionResult(observations, cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY), binary_view,
            [contour for _,_,contour in views], region_name="ROIs", score_name="ocr",
            diagnostics={"mode":"direct_roi" if self.direct else getattr(self.locator, "name", type(self.locator).__name__), "inputs":records,
                "timings_ms":{"geometry":geometry_ms,"text_localization":locate_ms,"ocr":read_ms,
                              "recovery":recovery_ms,
                              "total":(perf_counter()-started)*1000}}, debug_crops=crops)
