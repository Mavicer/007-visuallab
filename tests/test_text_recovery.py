"""Geometry, abstention and runtime-budget checks without model downloads."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from visuallab.detection import DetectionResult, Observation
from visuallab.text_recovery import (foreground_view, planar_text_view,
                                    recover_text, rotate_view, visible_paper_quads)
from visuallab.text_regions import RegionTextDetector, TextRegion
from visuallab.yolo import YoloTextLocator


class Reader:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return next(self.replies)


def foreground_fixture():
    gray = np.full((120,100),245,np.uint8)
    cv2.rectangle(gray,(40,20),(60,95),20,-1)
    return gray


class RecoveryTests(unittest.TestCase):
    def test_rotated_canvas_preserves_all_source_corners_and_inverse(self):
        gray = foreground_fixture()
        corners = np.array([[[0,0],[99,0],[99,119],[0,119]]],np.float32)
        for angle in (-90,-35,35,90):
            view = rotate_view(gray,angle)
            mapped = cv2.perspectiveTransform(corners,np.linalg.inv(view.to_source))
            self.assertGreaterEqual(float(mapped.min()),-1e-5)
            self.assertLessEqual(float(mapped[:,:,0].max()),view.image.shape[1]-1+1e-5)
            self.assertLessEqual(float(mapped[:,:,1].max()),view.image.shape[0]-1+1e-5)
            restored = cv2.perspectiveTransform(mapped,view.to_source)
            np.testing.assert_allclose(restored,corners,atol=1e-4)

    def test_foreground_keeps_disconnected_punctuation_and_rejects_border(self):
        gray = np.full((120,100),245,np.uint8)
        cv2.rectangle(gray,(0,0),(12,119),0,-1)
        cv2.circle(gray,(50,45),4,20,-1)
        cv2.circle(gray,(50,75),4,20,-1)
        view = foreground_view(gray)
        mask = (view.image < 100).astype(np.uint8)
        self.assertEqual(cv2.connectedComponents(mask)[0]-1,2)
        self.assertGreaterEqual(int(view.image.min()),15)

    def test_blank_abstains_without_calling_reader(self):
        reader = Reader([])
        view,_,_,diagnostic = recover_text(reader,np.full((80,100),245,np.uint8))
        self.assertIsNone(view)
        self.assertEqual(reader.calls,0)
        self.assertEqual(diagnostic['status'],'no_reliable_foreground')

    def test_competing_high_score_directions_abstain(self):
        reader = Reader([('A',.95),('B',.94),('A',.9)])
        view,_,_,diagnostic = recover_text(reader,foreground_fixture())
        self.assertIsNone(view)
        self.assertEqual(diagnostic['status'],'weak_or_ambiguous')
        self.assertEqual(reader.calls,3)

    def test_neighboring_high_score_disagreement_abstains(self):
        reader = Reader([('A',.95),('B',.2),('',0.),('A',.9),('B',.85)])
        view,_,_,diagnostic = recover_text(reader,foreground_fixture())
        self.assertIsNone(view)
        self.assertEqual(diagnostic['status'],'unstable_direction')
        self.assertLessEqual(reader.calls,6)

    def test_two_agreements_and_one_weak_neighbor_can_recover(self):
        reader = Reader([('A',.95),('B',.2),('',0.),('A',.9),('B',.4)])
        view,text,score,_ = recover_text(reader,foreground_fixture())
        self.assertIsNotNone(view)
        self.assertEqual(text,'A')
        self.assertEqual(score,.95)
        self.assertEqual(reader.calls,5)

    def test_real_perspective_corners_and_inverse_preserve_corner_identity(self):
        paper = np.full((240,180,3),245,np.uint8)
        source = np.array([[0,0],[179,0],[179,239],[0,239]],np.float32)
        carrier = np.array([[80,40],[240,80],[265,310],[30,260]],np.float32)
        text = np.array([[40,65],[140,65],[140,180],[40,180]],np.float32)
        for point,value in zip(text,(0,60,120,180)):
            cv2.circle(paper,tuple(point.astype(int)),12,(value,)*3,-1)
        matrix = cv2.getPerspectiveTransform(source,carrier)
        image = cv2.warpPerspective(paper,matrix,(320,350))
        projected = cv2.perspectiveTransform(text[None],matrix)[0]
        view = planar_text_view(image,projected,[carrier])
        self.assertIsNotNone(view)
        self.assertEqual(view.method,'paper_homography')
        for pixel,value in zip((view.image[2,2],view.image[2,-3],view.image[-3,-3],view.image[-3,2]),(0,60,120,180)):
            self.assertAlmostEqual(int(pixel),value,delta=12)
        self.assertIsNone(planar_text_view(image,projected,[]))

    def test_visible_carrier_provider_finds_complete_quad_but_rejects_clipping(self):
        image = np.full((350,350,3),80,np.uint8)
        quad = np.array([[90,40],[260,70],[300,300],[40,280]],np.int32)
        cv2.fillConvexPoly(image,quad,(245,245,245))
        cv2.putText(image,'AB',(115,190),cv2.FONT_HERSHEY_SIMPLEX,1.5,(10,10,10),4)
        candidates = visible_paper_quads(image)
        self.assertTrue(candidates)
        self.assertTrue(any(np.max(np.linalg.norm(candidate-quad,axis=1))<6 for candidate in candidates))
        # Crop through a physical corner: a frame edge is not a paper corner.
        self.assertEqual(visible_paper_quads(image[:,100:]),[])

    def test_accepted_primary_read_never_calls_recovery(self):
        reader = Reader([('A',.9),('B',.99)])
        with patch('visuallab.text_recovery.recover_text') as recover:
            result = RegionTextDetector(direct=True,reader=reader,recovery=True).detect(
                cv2.cvtColor(foreground_fixture(),cv2.COLOR_GRAY2BGR))
        recover.assert_not_called()
        self.assertEqual(result.observations[0].label,'A')

    def test_only_two_unreadable_regions_get_extra_work(self):
        class Locator:
            def locate(self,image):
                return [TextRegion(np.array([[x,20],[x+20,20],[x+20,80],[x,80]],np.float32))
                        for x in (20,70,120)]
        reader = Reader([('',0.)]*6)
        with patch('visuallab.text_recovery.visible_paper_quads',return_value=[]) as geometry, \
             patch('visuallab.text_recovery.recover_text',return_value=(None,'',0.,{'status':'unknown'})) as recover:
            result = RegionTextDetector(locator=Locator(),reader=reader,recovery=True).detect(
                np.full((110,180,3),245,np.uint8))
        self.assertEqual(recover.call_count,2)
        self.assertEqual(geometry.call_count,2)
        self.assertEqual(result.diagnostics['inputs'][2]['recovery']['status'],'frame_budget_exhausted')

    def test_rotated_recovery_binary_projects_to_original_pixels(self):
        gray = foreground_fixture()
        rotated = rotate_view(gray,90)
        reader = Reader([('',0.),('A',.99),('X',.9)])
        with patch('visuallab.text_recovery.visible_paper_quads',return_value=[]), \
             patch('visuallab.text_recovery.recover_text',return_value=(rotated,'A',.95,{'status':'recovered'})):
            result = RegionTextDetector(direct=True,reader=reader,recovery=True).detect(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))
        expected = gray < 100
        actual = result.binary > 0
        self.assertGreater(np.count_nonzero(actual & expected)/np.count_nonzero(actual | expected),.98)
        self.assertEqual(result.observations[0].box,(0,0,100,120))

    def test_empty_detector_retry_maps_coordinates_back_without_rotating_labels(self):
        class Detector:
            class_names = {0:'text'}
            def __init__(self): self.shapes = []
            def detect(self,image):
                self.shapes.append(image.shape)
                items = [] if len(self.shapes)==1 else [Observation((10,20,30,40),(25,40),'text',.9)]
                return DetectionResult(items)
        detector = Detector()
        regions = YoloTextLocator(detector,recover_empty=True).locate(np.zeros((80,140,3),np.uint8))
        self.assertEqual(detector.shapes,[(80,140,3),(140,80,3)])
        self.assertEqual(cv2.boundingRect(regions[0].quad),(20,40,40,30))
        self.assertEqual(regions[0].origin,'rotated_detection_90')

    def test_detector_does_not_retry_when_primary_proposal_exists(self):
        class Detector:
            class_names = {0:'text'}
            calls = 0
            def detect(self,image):
                self.calls += 1
                return DetectionResult([Observation((10,20,30,40),(25,40),'text',.9)])
        detector = Detector()
        YoloTextLocator(detector,recover_empty=True).locate(np.zeros((80,140,3),np.uint8))
        self.assertEqual(detector.calls,1)


if __name__ == '__main__':
    unittest.main()
