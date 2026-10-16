"""Check the hybrid boundary without downloading weights or opening a camera."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from visuallab.yolo import HybridCardDetector, YoloObjectDetector, YoloTextLocator


class FakeModel:
    def __init__(self, names, boxes=(), confidences=(), classes=()):
        self.names = names
        self.boxes = SimpleNamespace(
            xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
            conf=np.asarray(confidences, dtype=float),
            cls=np.asarray(classes, dtype=float),
        )

    def predict(self, **kwargs):
        assert kwargs["source"].ndim == 3
        return [SimpleNamespace(boxes=self.boxes)]


class YoloAdapterTests(unittest.TestCase):
    def test_text_proposals_are_deduplicated_before_reading(self):
        model = FakeModel({0:"text"}, [[10,10,80,90],[12,12,82,92],[130,10,180,90]],
                          [.9,.8,.7], [0,0,0])
        regions = YoloTextLocator(YoloObjectDetector("injected.pt",model=model)).locate(
            np.zeros((120,200,3),np.uint8))
        self.assertEqual(len(regions),2)
        self.assertEqual([r.score for r in regions],[.9,.7])

    def test_text_locator_rejects_carrier_weights_and_preserves_text_coordinates(self):
        with self.assertRaisesRegex(ValueError, "必须包含 'text' 类"):
            YoloTextLocator(YoloObjectDetector("injected.pt", model=FakeModel({0:"card"})))
        model = FakeModel({0:"text",1:"person"}, [[10,20,60,90],[0,0,100,100]], [.9,.8], [0,1])
        regions = YoloTextLocator(YoloObjectDetector("injected.pt",model=model)).locate(
            np.zeros((120,160,3),np.uint8))
        self.assertEqual(len(regions),1)
        np.testing.assert_array_equal(regions[0].quad, [[10,20],[59,20],[59,89],[10,89]])
        self.assertEqual(regions[0].score,.9)

    def test_coordinates_are_clipped_and_classes_are_generic(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        model = FakeModel({0: "person", 1: "cat"},
                          [[-8, 10, 61.2, 95], [100, 20, 210, 130]],
                          [0.83, 0.67], [0, 1])
        result = YoloObjectDetector("injected.pt", model=model).detect(frame)
        self.assertEqual([item.box for item in result.observations],
                         [(0, 10, 62, 85), (100, 20, 100, 100)])
        self.assertEqual([item.label for item in result.observations], ["person", "cat"])
        self.assertEqual([item.center for item in result.observations], [(31, 52), (150, 70)])

    def test_hybrid_rejects_coco_and_no_card_skips_paper_search(self):
        stock = YoloObjectDetector("injected.pt", model=FakeModel({0: "person"}))
        with self.assertRaisesRegex(ValueError, "没有 'card' 类"):
            HybridCardDetector(stock)
        card = YoloObjectDetector("injected.pt", model=FakeModel({0: "card"}))
        with patch("visuallab.processing._paper_regions") as search:
            result = HybridCardDetector(card).detect(
                np.full((200, 300, 3), 125, dtype=np.uint8)
            )
        search.assert_not_called()
        self.assertEqual(result.observations, [])
        self.assertEqual(len(result.regions), 0)
        self.assertEqual(np.count_nonzero(result.binary), 0)

    def test_two_detected_cards_are_read_inside_their_rois(self):
        frame = np.full((480, 800, 3), (70, 100, 140), dtype=np.uint8)
        for left, digit in ((50, "1"), (440, "2")):
            cv2.rectangle(frame, (left, 80), (left + 300, 420), (240, 240, 240), -1)
            cv2.putText(frame, digit, (left + 100, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (10, 10, 10), 16, cv2.LINE_AA)
        model = FakeModel({0: "card"},
                          [[50, 80, 351, 421], [440, 80, 741, 421]],
                          [.9, .8], [0, 0])
        result = HybridCardDetector(
            YoloObjectDetector("injected.pt", model=model)
        ).detect(frame)
        self.assertEqual([item.label for item in result.observations], ["1", "2"])
        self.assertEqual(len(result.regions), 2)
        self.assertTrue(all(item.polygon is not None for item in result.observations))
        self.assertEqual([item.source_score for item in result.observations], [.9, .8])


if __name__ == "__main__":
    unittest.main()
