"""Text ROI boundaries and decisions; fake inference keeps these offline."""
import unittest

import cv2
import numpy as np

from visuallab.text_regions import DBTextLocator, RegionTextDetector, TextRegion, crop_quad


class RecordingReader:
    def __init__(self, replies=None):
        self.inputs = []
        self.replies = iter(replies or [('?', .95), ('?', .98)])

    def predict(self, image):
        self.inputs.append(image.copy())
        return next(self.replies)


class FixedLocator:
    def __init__(self):
        self.calls = []

    def locate(self, image):
        self.calls.append(image.copy())
        return [TextRegion(np.array([[20,10],[80,10],[80,70],[20,70]], np.float32), .9)]


class TextRegionTests(unittest.TestCase):
    def test_direct_roi_preserves_disconnected_punctuation_and_question_mark(self):
        image = np.full((120, 240, 3), 245, np.uint8)
        cv2.circle(image, (95, 50), 2, (0,0,0), -1)
        cv2.circle(image, (95, 65), 2, (0,0,0), -1)
        # An exterior background mark must never enter the recognizer.
        cv2.rectangle(image, (0,0), (25,119), (0,0,0), -1)
        reader = RecordingReader()
        result = RegionTextDetector(direct=True, reader=reader,
                    target_roi=(.25,.25,.5,.5)).detect(image)
        self.assertEqual(reader.inputs[0].shape, (60,120,3))
        self.assertEqual(cv2.connectedComponents((reader.inputs[0][:,:,0] < 100).astype(np.uint8))[0]-1, 2)
        self.assertEqual(result.observations[0].label, '?')
        self.assertFalse(result.observations[0].is_unknown)
        self.assertEqual(result.observations[0].box, (60,30,120,60))

    def test_higher_binary_confidence_never_overrides_gray(self):
        reader = RecordingReader([('2', .7), ('2-', .99)])
        result = RegionTextDetector(direct=True, reader=reader).detect(np.full((90,90,3),200,np.uint8))
        self.assertEqual(result.observations[0].label, '2')
        self.assertEqual(result.observations[0].score, .7)
        self.assertTrue(result.diagnostics['inputs'][0]['disagreement'])
        self.assertEqual(len(result.debug_crops), 2)

    def test_binary_does_not_rescue_a_low_score_gray_prediction(self):
        reader = RecordingReader([('I', .4), ('1', .99)])
        result = RegionTextDetector(direct=True, reader=reader).detect(np.full((90,90,3),200,np.uint8))
        self.assertTrue(result.observations[0].is_unknown)
        self.assertEqual(result.diagnostics['inputs'][0]['status'], 'unknown')

    def test_empty_object_proposals_never_search_background(self):
        locator = FixedLocator()
        reader = RecordingReader()
        result = RegionTextDetector(locator=locator, reader=reader).detect_in_boxes(np.zeros((100,100,3),np.uint8), [])
        self.assertEqual(locator.calls, [])
        self.assertEqual(reader.inputs, [])
        self.assertEqual(result.observations, [])

    def test_text_coordinates_map_back_to_object_and_frame(self):
        locator = FixedLocator()
        result = RegionTextDetector(locator=locator, reader=RecordingReader()).detect_in_boxes(
            np.full((300,400,3),200,np.uint8), [(100,80,150,120)])
        self.assertEqual(locator.calls[0].shape, (120,150,3))
        polygon = result.observations[0].polygon
        self.assertTrue((polygon[:,0] >= 100).all())
        self.assertTrue((polygon[:,1] >= 80).all())
        np.testing.assert_allclose(polygon.mean(axis=0), [150,120], atol=1)

    def test_true_perspective_quad_preserves_four_corner_colors(self):
        image = np.full((150,190,3),255,np.uint8)
        quad = np.array([[30,20],[160,40],[145,125],[20,105]], np.float32)
        colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0)]
        for point,color in zip(quad,colors):
            cv2.circle(image,tuple(point.astype(int)),7,color,-1)
        crop = crop_quad(image,quad)
        for pixel,color in zip([crop[0,0],crop[0,-1],crop[-1,-1],crop[-1,0]],colors):
            np.testing.assert_allclose(pixel,color,atol=2)

    def test_db_map_scales_regions_and_rejects_weak_or_tiny_noise(self):
        probability = np.zeros((64,96), np.float32)
        probability[20:35,30:60] = .95
        probability[45:57,5:15] = .4
        probability[1:2,1:2] = .99
        regions = DBTextLocator.regions_from_map(probability,(128,192))
        self.assertEqual(len(regions),1)
        np.testing.assert_allclose(regions[0].quad.mean(axis=0),[89,54],atol=1)
        self.assertGreater(regions[0].score,.9)

    def test_invalid_or_inverted_roi_is_rejected(self):
        for roi in ((.9,.1,.2,.2),(0,0,0,.2),(-.1,0,.5,.5),(0,0,float('nan'),.5)):
            with self.assertRaises(ValueError):
                RegionTextDetector(direct=True,reader=RecordingReader(),target_roi=roi)


if __name__ == '__main__':
    unittest.main()
