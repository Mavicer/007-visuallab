"""Offline checks for the image pipeline; no camera or pytest required."""

import unittest
from pathlib import Path

import cv2
import numpy as np

from visuallab.processing import process_frame
from visuallab.recognition import DigitRecognizer
from visuallab.detection import NumberCardDetector


def recognized_labels(frame: np.ndarray, mode: str = "adaptive") -> list[str]:
    """Run segmentation and recognition together, as the camera loop does."""
    analysis = process_frame(frame, mode)
    recognizer = DigitRecognizer()
    found = []
    for candidate in analysis.candidates:
        label, _ = recognizer.predict(candidate.mask)
        if label != "?":
            found.append((candidate.box[0], label))
    return [label for _, label in sorted(found)]


class PipelineTests(unittest.TestCase):
    def test_two_cards_on_light_and_dark_backgrounds(self) -> None:
        for background in (120, 180, 235):
            with self.subTest(background=background):
                frame = np.full((480, 800, 3), background, dtype=np.uint8)
                for left, digit in ((50, "1"), (440, "2")):
                    cv2.rectangle(frame, (left, 80), (left + 300, 420), (250, 250, 250), -1)
                    cv2.putText(
                        frame, digit, (left + 105, 305), cv2.FONT_HERSHEY_SIMPLEX,
                        5, (10, 10, 10), 12, cv2.LINE_AA
                    )
                self.assertEqual(recognized_labels(frame), ["1", "2"])

    def test_different_font_and_size(self) -> None:
        for font, scale in ((cv2.FONT_HERSHEY_COMPLEX, 4), (cv2.FONT_HERSHEY_PLAIN, 5)):
            for digit in ("1", "2"):
                with self.subTest(font=font, digit=digit):
                    frame = np.full((360, 500, 3), 150, dtype=np.uint8)
                    cv2.rectangle(frame, (100, 35), (400, 330), (250, 250, 250), -1)
                    cv2.putText(
                        frame, digit, (170, 265), font, scale,
                        (15, 15, 15), 9, cv2.LINE_AA
                    )
                    self.assertEqual(recognized_labels(frame), [digit])

    def test_uneven_light_and_blur(self) -> None:
        gradient = np.linspace(155, 235, 800, dtype=np.uint8)
        frame = np.repeat(gradient[np.newaxis, :, np.newaxis], 480, axis=0)
        frame = np.repeat(frame, 3, axis=2)
        for left, digit in ((50, "1"), (440, "2")):
            cv2.rectangle(frame, (left, 80), (left + 300, 420), (248, 248, 248), -1)
            cv2.putText(frame, digit, (left + 105, 305), cv2.FONT_HERSHEY_SIMPLEX,
                        5, (20, 20, 20), 12, cv2.LINE_AA)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        self.assertEqual(recognized_labels(frame), ["1", "2"])

    def test_blank_and_solid_block_are_not_digits(self) -> None:
        frame = np.full((400, 600, 3), 120, dtype=np.uint8)
        cv2.rectangle(frame, (130, 50), (470, 350), (245, 245, 245), -1)
        self.assertEqual(recognized_labels(frame), [])
        cv2.rectangle(frame, (180, 100), (220, 290), (10, 10, 10), -1)
        self.assertEqual(recognized_labels(frame), [])

    def test_partial_occlusion_does_not_change_two_into_one(self) -> None:
        frame = np.full((400, 400, 3), 120, dtype=np.uint8)
        cv2.rectangle(frame, (70, 35), (330, 365), (235, 235, 235), -1)
        cv2.putText(
            frame, "2", (120, 300), cv2.FONT_HERSHEY_SIMPLEX,
            5, (10, 10, 10), 12, cv2.LINE_AA
        )
        cv2.rectangle(frame, (100, 220), (270, 248), (235, 235, 235), -1)
        self.assertNotIn("1", recognized_labels(frame))

    def test_clutter_without_paper_is_not_recognized(self) -> None:
        frame = np.full((480, 800, 3), (110, 145, 185), dtype=np.uint8)
        cv2.rectangle(frame, (30, 35), (270, 220), (35, 45, 60), 15)
        cv2.line(frame, (450, 40), (510, 300), (20, 25, 30), 18)
        cv2.circle(frame, (610, 270), 85, (25, 35, 45), 12)
        self.assertEqual(recognized_labels(frame), [])

    def test_real_card_shapes_on_cluttered_background(self) -> None:
        frame = np.full((480, 800, 3), (90, 130, 175), dtype=np.uint8)
        cv2.rectangle(frame, (10, 15), (170, 170), (25, 35, 45), 14)
        # Adjacent cards may become one white surface; both digits must survive.
        for left, digit in ((180, "1"), (410, "2")):
            cv2.rectangle(frame, (left, 90), (left + 230, 390), (220, 220, 220), -1)
            sample = cv2.imread(
                str(Path(__file__).resolve().parents[1]
                    / "visuallab" / "templates" / f"{digit}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertIsNotNone(sample)
            enlarged = cv2.resize(sample, (144, 144), interpolation=cv2.INTER_LINEAR)
            roi = frame[160:304, left + 43:left + 187]
            roi[enlarged > 100] = (20, 20, 20)
        self.assertEqual(recognized_labels(frame), ["1", "2"])

    def test_card_rotated_in_camera_plane(self) -> None:
        sample = cv2.imread(
            str(Path(__file__).resolve().parents[1] / "visuallab" / "templates" / "1.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(sample)
        card = np.full((300, 220, 3), 220, dtype=np.uint8)
        enlarged = cv2.resize(sample, (144, 144), interpolation=cv2.INTER_LINEAR)
        card[80:224, 38:182][enlarged > 100] = (20, 20, 20)
        for angle in (10, 20, 30):
            with self.subTest(angle=angle):
                frame = np.full((480, 640, 3), (100, 140, 170), dtype=np.uint8)
                rotation = cv2.getRotationMatrix2D((110, 150), angle, 1)
                turned = cv2.warpAffine(card, rotation, (220, 300))
                coverage = cv2.warpAffine(
                    np.full(card.shape[:2], 255, dtype=np.uint8), rotation, (220, 300)
                )
                roi = frame[90:390, 210:430]
                roi[coverage > 0] = turned[coverage > 0]
                self.assertEqual(recognized_labels(frame), ["1"])

    def test_thin_paper_edge_joined_to_digit_is_separated(self) -> None:
        recognizer = DigitRecognizer()
        for digit, line_x in (("1", 290), ("2", 295)):
            with self.subTest(digit=digit):
                frame = np.full((480, 640, 3), (100, 140, 175), dtype=np.uint8)
                cv2.rectangle(frame, (140, 50), (460, 420), (210, 210, 210), -1)
                cv2.putText(
                    frame, digit, (250, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    5, (20, 20, 20), 12, cv2.LINE_AA,
                )
                # A narrow dark card seam reaches and joins the digit's top.
                cv2.line(frame, (line_x, 50), (line_x, 235), (70, 70, 70), 3)
                candidates = process_frame(frame).candidates
                self.assertEqual(len(candidates), 1)
                self.assertGreater(candidates[0].box[1], 200)
                self.assertEqual(recognizer.predict(candidates[0].mask)[0], digit)

    def test_warm_paper_and_thick_ink_keep_filled_digits(self) -> None:
        frame = np.full((480, 800, 3), (65, 85, 105), dtype=np.uint8)
        for left, color, digit in (
            (50, (225, 225, 225), "1"),
            (440, (185, 205, 225), "2"),
        ):
            cv2.rectangle(frame, (left, 70), (left + 300, 430), color, -1)
            cv2.putText(
                frame, digit, (left + 85, 320), cv2.FONT_HERSHEY_SIMPLEX,
                5, (20, 20, 30), 26, cv2.LINE_AA,
            )
        analysis = process_frame(frame)
        self.assertEqual(recognized_labels(frame), ["1", "2"])
        # The displayed binary view contains solid candidate strokes, not
        # the hollow outlines produced by the old 31-pixel threshold window.
        self.assertGreater(np.count_nonzero(analysis.binary[160:320, 130:290]), 1500)
        self.assertEqual(np.count_nonzero(analysis.binary[:50, :]), 0)

    def test_strict_paper_proposal_survives_bright_neighbor(self) -> None:
        frame = np.full((420, 760, 3), (55, 75, 95), dtype=np.uint8)
        cv2.rectangle(frame, (80, 60), (350, 370), (240, 240, 240), -1)
        # This bright warm object touches the card. A relaxed color mask
        # merges them, while the strict mask should still isolate the card.
        cv2.rectangle(frame, (351, 80), (690, 350), (180, 200, 220), -1)
        cv2.putText(
            frame, "1", (160, 300), cv2.FONT_HERSHEY_SIMPLEX,
            4, (20, 20, 20), 18, cv2.LINE_AA,
        )
        self.assertEqual(recognized_labels(frame), ["1"])

    def test_perspective_tilt_rectifies_two_cards(self) -> None:
        frame = np.full((720, 1200, 3), (75, 95, 120), dtype=np.uint8)
        source = np.array([[0, 0], [259, 0], [259, 359], [0, 359]], dtype=np.float32)
        cards = (
            ("1", (235, 235, 235), [[80, 160], [420, 70], [365, 650], [105, 600]]),
            ("2", (190, 205, 225), [[650, 80], [1070, 140], [1110, 610], [585, 670]]),
        )
        for digit, color, corners in cards:
            card = np.full((360, 260, 3), color, dtype=np.uint8)
            cv2.putText(
                card, digit, (68, 285), cv2.FONT_HERSHEY_SIMPLEX,
                5, (20, 20, 20), 20, cv2.LINE_AA,
            )
            transform = cv2.getPerspectiveTransform(source, np.array(corners, dtype=np.float32))
            warped = cv2.warpPerspective(card, transform, (1200, 720))
            coverage = cv2.warpPerspective(
                np.full((360, 260), 255, dtype=np.uint8), transform, (1200, 720)
            )
            frame[coverage > 0] = warped[coverage > 0]

        for mode in ("adaptive", "otsu"):
            with self.subTest(mode=mode):
                observations = NumberCardDetector(mode).detect(frame).observations
                self.assertEqual([o.label for o in observations], ["1", "2"])
                self.assertTrue(all(o.polygon is not None for o in observations))
                self.assertTrue(all(o.center[0] > 0 and o.center[1] > 0 for o in observations))


if __name__ == "__main__":
    unittest.main()
