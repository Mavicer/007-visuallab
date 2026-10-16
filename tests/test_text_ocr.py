"""OCR boundary checks, requiring no model download or ONNX installation."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from visuallab.text_ocr import CardTextDetector, decode_ctc
from visuallab.processing import _paper_proposals_from_boxes


class Reader:
    def __init__(self, text):
        self.text = text
        self.inputs = []

    def predict(self, image):
        self.inputs.append(image)
        return self.text, .95


class TextOcrTests(unittest.TestCase):
    def test_ctc_preserves_punctuation_and_repeat_after_blank(self):
        alphabet = ['', 'A', '?', ' ']
        tokens = [1, 1, 0, 1, 3, 2, 2]
        probabilities = np.full((len(tokens), len(alphabet)), .01)
        for row, token in enumerate(tokens):
            probabilities[row, token] = .97
        text, score = decode_ctc(probabilities, alphabet)
        self.assertEqual(text, 'AA ?')
        self.assertGreater(score, .9)

    def test_small_disconnected_punctuation_reaches_reader(self):
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        cv2.rectangle(frame, (50, 30), (270, 210), (245, 245, 245), -1)
        # A colon is two tiny components, rejected by the old digit filters.
        cv2.circle(frame, (150, 100), 3, (10, 10, 10), -1)
        cv2.circle(frame, (150, 130), 3, (10, 10, 10), -1)
        reader = Reader(':')
        contour = np.array([[[50, 30]], [[270, 30]], [[270, 210]], [[50, 210]]], dtype=np.int32)
        with patch('visuallab.text_ocr.locate_papers', return_value=[(contour, False)]):
            result = CardTextDetector(reader=reader).detect(frame)
        self.assertEqual(len(reader.inputs), 1)
        self.assertEqual(result.observations[0].label, ':')
        self.assertFalse(result.observations[0].is_unknown)
        count = cv2.connectedComponents((result.binary > 0).astype(np.uint8))[0] - 1
        self.assertEqual(count, 2)

    def test_literal_question_mark_is_not_an_unknown_result(self):
        frame = np.full((200, 300, 3), 240, dtype=np.uint8)
        cv2.putText(frame, '?', (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
        contour = np.array([[[0, 0]], [[299, 0]], [[299, 199]], [[0, 199]]], dtype=np.int32)
        with patch('visuallab.text_ocr.locate_papers', return_value=[(contour, False)]):
            result = CardTextDetector(reader=Reader('?')).detect(frame)
        self.assertEqual(result.observations[0].label, '?')
        self.assertFalse(result.observations[0].is_unknown)

    def test_empty_yolo_boxes_do_not_trigger_full_frame_search(self):
        reader = Reader('unexpected')
        with patch('visuallab.processing._paper_regions') as search:
            result = CardTextDetector(reader=reader).detect_in_boxes(
                np.zeros((200, 300, 3), np.uint8), [])
        search.assert_not_called()
        self.assertEqual(reader.inputs, [])
        self.assertEqual(result.observations, [])

    def test_occluded_paper_keeps_its_mask_without_fake_perspective(self):
        frame = np.full((400, 500, 3), 180, np.uint8)
        # A clipped octagon cannot provide four true paper corners. It is
        # still better than including all background inside the YOLO AABB.
        local = np.array([[[124, 44]], [[204, 44]], [[244, 104]], [[244, 204]],
                          [[184, 244]], [[64, 244]], [[44, 164]], [[44, 104]]], np.int32)
        with patch('visuallab.processing._paper_regions', return_value=[local]):
            proposals = _paper_proposals_from_boxes(
                frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), [(100, 100, 200, 200)])
        self.assertEqual(len(proposals), 1)
        contour, can_rectify = proposals[0]
        self.assertEqual(len(contour), 8)
        self.assertFalse(can_rectify)


if __name__ == '__main__':
    unittest.main()
