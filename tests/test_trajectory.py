"""Visual-trail association, loss handling and raw-frame isolation."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from visuallab.camera import run_preview
from visuallab.detection import DetectionResult, Observation
from visuallab.terminal_status import PipelineInfo,TerminalDashboard
from visuallab.trajectory import TrajectoryTrail


SHAPE=(240,400,3)


def subject(x=80,y=100,label='+',unknown=False,score=.95):
    return Observation((x-20,y-20,40,40),(x,y),label,score,is_unknown=unknown)


class TrajectoryTests(unittest.TestCase):
    def test_only_accepted_plus_can_acquire(self):
        effect=TrajectoryTrail()
        for item in [subject(label='2'),subject(label='x'),subject(label='*'),subject(unknown=True)]:
            effect.update([item],SHAPE,0.)
            self.assertFalse(effect.points)
        effect.update([subject()],SHAPE,.1)
        self.assertEqual(effect.points[-1].center,(80,100))

    def test_point_history_is_bounded_and_expires_by_time(self):
        effect=TrajectoryTrail()
        for i in range(300):
            effect.update([subject(80+i%10)],SHAPE,i/1000)
        self.assertLessEqual(len(effect.points),128)
        effect.update([],SHAPE,2.)
        self.assertFalse(effect.points)

    def test_short_unknown_coast_uses_real_centers_but_does_not_renew_confirmation(self):
        effect=TrajectoryTrail()
        effect.update([subject()],SHAPE,0.)
        effect.update([subject(90,unknown=True)],SHAPE,.1)
        self.assertEqual(effect.points[-1].center,(90,100))
        effect.update([subject(95,unknown=True)],SHAPE,.41)
        self.assertEqual(len(effect.points),2)
        effect.update([subject(95,label='4')],SHAPE,.42)
        self.assertEqual(len(effect.points),2)

    def test_missing_frame_lifts_pen_and_far_reacquisition_never_bridges(self):
        effect=TrajectoryTrail()
        effect.update([subject()],SHAPE,0.)
        effect.update([],SHAPE,.05)
        effect.update([subject(95)],SHAPE,.1)
        self.assertTrue(effect.points[-1].start)
        effect.update([subject(360)],SHAPE,.2)
        self.assertEqual(effect.points[-1].center,(95,100))
        effect.update([subject(360)],SHAPE,.7)
        self.assertEqual(len(effect.points),1)
        self.assertTrue(effect.points[0].start)
        self.assertEqual(effect.points[0].center,(360,100))

    def test_ambiguous_targets_pause_but_overlapping_duplicates_are_one(self):
        effect=TrajectoryTrail()
        effect.update([subject(80),subject(180)],SHAPE,0.)
        self.assertFalse(effect.points)
        effect.update([subject(80),subject(82,score=.8)],SHAPE,.1)
        self.assertEqual(len(effect.points),1)
        effect.update([subject(70),subject(120)],SHAPE,.2)
        self.assertEqual(len(effect.points),1)

    def test_resolution_clock_change_and_disable_clear_state(self):
        effect=TrajectoryTrail()
        effect.update([subject()],SHAPE,1.)
        effect.update([subject(90)],(480,800,3),1.1)
        self.assertEqual(len(effect.points),1)
        effect.update([subject(95)],(480,800,3),.5)
        self.assertEqual(len(effect.points),1)
        effect.enabled=False
        effect.update([subject()],SHAPE,.6)
        self.assertFalse(effect.points)

    def test_draw_fades_and_clips_glow_without_modifying_observations(self):
        effect=TrajectoryTrail()
        items=[subject(30,30),subject(80,40),subject(130,50)]
        for i,item in enumerate(items):effect.update([item],SHAPE,.1*i)
        effect.update([],SHAPE,.25)
        early=np.zeros(SHAPE,np.uint8);late=early.copy();expired=early.copy()
        effect.draw(early,.25);effect.draw(late,1.2);effect.draw(expired,2.)
        self.assertGreater(early.sum(),late.sum())
        self.assertGreater(late.sum(),0)
        self.assertEqual(expired.sum(),0)
        self.assertEqual(items[-1].center,(130,50))
        edge=TrajectoryTrail();edge.update([subject(1,1)],SHAPE,0.)
        edge.draw(np.zeros(SHAPE,np.uint8),0.)  # No out-of-frame ROI errors.

    def test_live_controls_and_save_keep_original_frames_and_callback_coordinates(self):
        frame=np.full(SHAPE,120,np.uint8);original=frame.copy()
        gray=np.full(SHAPE[:2],77,np.uint8);binary=np.zeros(SHAPE[:2],np.uint8)
        class Source:
            released=False
            def read(self):return True,frame
            def release(self):self.released=True
        class Detector:
            count=0
            def detect(self,image):
                self.count+=1
                return DetectionResult([subject(70+10*self.count)],gray,binary)
        source,detector=Source(),Detector();callbacks=[];previews=[]
        from visuallab.preview_layout import compose_preview
        def capture_preview(g,b,marked,**kwargs):
            np.testing.assert_array_equal(g,gray)
            np.testing.assert_array_equal(b,binary)
            previews.append(marked.copy())
            return compose_preview(g,b,marked,**kwargs)
        with patch('visuallab.camera.cv2.namedWindow'),patch('visuallab.camera.cv2.resizeWindow'), \
             patch('visuallab.camera.cv2.moveWindow'),patch('visuallab.camera.cv2.imshow'), \
             patch('visuallab.camera.cv2.destroyAllWindows'), \
             patch('visuallab.camera.cv2.waitKey',side_effect=[0,ord('t'),0,ord('t'),ord('c'),ord('s'),ord('q')]), \
             patch('visuallab.camera.Path.mkdir'), \
             patch('visuallab.camera.cv2.imwrite',return_value=True) as save, \
             patch('visuallab.camera.compose_preview',side_effect=capture_preview):
            run_preview(source=source,detector=detector,
                status=TerminalDashboard(PipelineInfo('fake'),mode='off'),
                on_result=lambda result,size:callbacks.append((result.observations[0].center,size)))
        self.assertTrue(source.released)
        self.assertEqual(len(previews),7)
        self.assertEqual(callbacks[-1],((140,100),(400,240)))
        np.testing.assert_array_equal(frame,original)
        np.testing.assert_array_equal(save.call_args.args[1],original)
        # T disabled the effect for frame 3: its center retains source color.
        np.testing.assert_array_equal(previews[2][100,100],original[100,100])
        # T enabled it again before frame 5: a visible head modifies the center.
        self.assertFalse(np.array_equal(previews[4][100,120],original[100,120]))


if __name__=='__main__':unittest.main()
