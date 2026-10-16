"""Small, local template recognizer for the user's 1 and 2 cards."""

from pathlib import Path
import cv2
import numpy as np

IMAGE_SIZE = 48
INNER_SIZE = 38


def normalize_glyph(mask: np.ndarray) -> np.ndarray:
    """Crop a white glyph and center it at a fixed size without stretching it."""
    points = cv2.findNonZero(mask)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if points is None:
        return canvas

    x, y, width, height = cv2.boundingRect(points)
    glyph = mask[y:y + height, x:x + width]
    scale = INNER_SIZE / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(glyph, (new_width, new_height), interpolation=cv2.INTER_AREA)
    left = (IMAGE_SIZE - new_width) // 2
    top = (IMAGE_SIZE - new_height) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
    return canvas


def _make_templates() -> dict[str, list[np.ndarray]]:
    """Use real card samples first, then retain generic print-style examples."""
    templates: dict[str, list[np.ndarray]] = {"1": [], "2": []}
    sample_dir = Path(__file__).with_name("templates")
    for digit in templates:
        sample = cv2.imread(str(sample_dir / f"{digit}.png"), cv2.IMREAD_GRAYSCALE)
        if sample is not None:
            # Bundled masks contain a white stroke on a black background.
            sample = normalize_glyph(sample)
            for angle in (-30, -20, -10, 0, 10, 20, 30):
                rotation = cv2.getRotationMatrix2D(
                    (IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, 1
                )
                turned = cv2.warpAffine(
                    sample, rotation, (IMAGE_SIZE, IMAGE_SIZE),
                    flags=cv2.INTER_LINEAR
                )
                templates[digit].append(normalize_glyph(turned))
    fonts = (
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    )
    for digit in templates:
        for font in fonts:
            for thickness in (2, 3, 4):
                image = np.zeros((112, 112), dtype=np.uint8)
                (width, height), baseline = cv2.getTextSize(digit, font, 2.5, thickness)
                origin = ((112 - width) // 2, (112 + height - baseline) // 2)
                cv2.putText(
                    image, digit, origin, font, 2.5, 255, thickness, cv2.LINE_AA
                )
                templates[digit].append(normalize_glyph(image))
    return templates


class DigitRecognizer:
    """Compare a paper-contained stroke against normalized 1/2 templates."""

    def __init__(self, min_similarity: float = 0.50, min_margin: float = 0.08) -> None:
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.templates = _make_templates()

    def predict(self, mask: np.ndarray) -> tuple[str, float]:
        """Return a digit or '?' and its best template similarity (0 to 1)."""
        if mask.ndim != 2 or min(mask.shape) == 0 or mask.shape[0] < 20:
            return "?", 0.0

        # 实心黑块不是字形；先排除这类容易被误认为粗体「1」的区域。
        fill_ratio = np.count_nonzero(mask) / mask.size
        if fill_ratio > 0.78:
            return "?", 0.0

        glyph = normalize_glyph(mask) > 0
        if not np.any(glyph):
            return "?", 0.0

        scores: dict[str, float] = {}
        for digit, templates in self.templates.items():
            best = 0.0
            for template in templates:
                reference = template > 0
                intersection = np.count_nonzero(glyph & reference)
                union = np.count_nonzero(glyph | reference)
                best = max(best, intersection / union)
            scores[digit] = best

        label = max(scores, key=scores.get)
        best = scores[label]
        second = max(score for digit, score in scores.items() if digit != label)
        # 不够像任何模板，或两个数字分数太接近，就保留“未知”。
        if best < self.min_similarity or best - second < self.min_margin:
            return "?", float(best)
        return label, float(best)
