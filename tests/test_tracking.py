"""Track loss/ambiguity, image-measured flow and model-boundary regression."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from visuallab.detection import DetectionResult, Observation
from visuallab.tracking import TrackedObjectDetector
from visuallab.trajectory import TrajectoryTrail


def obj(x=150,y=120,label='armor'):
    return Observation((x-45,y-40,90,80),(x,y),label,.9)


class SequenceDetector:
    class_names={0:'armor'}
    def __init__(self,frames):self.frames=iter(frames);self.calls=0
    def detect(self,image):
        self.calls+=1
        return DetectionResult(next(self.frames))


class TrackingTests(unittest.TestCase):
    def setUp(self):self.frame=np.full((300,500,3),100,np.uint8)

    def test_model_class_validation_accepts_mapping_and_rejects_text_weights(self):
        TrackedObjectDetector(SequenceDetector([]))
        detector=SequenceDetector([]);detector.class_names={0:'text'}
        with self.assertRaisesRegex(ValueError,'整板'):TrackedObjectDetector(detector)

    def test_detector_runs_once_and_fast_motion_is_not_smoothed_backwards(self):
        detector=SequenceDetector([[obj()],[obj(320)]])
        tracker=TrackedObjectDetector(detector)
        tracker.detect_at(self.frame,0.)
        result=tracker.detect_at(self.frame,.05)
        self.assertEqual(result.observations[0].center,(320,120))
        self.assertEqual(result.diagnostics['tracking']['identity'],1)
        self.assertEqual(detector.calls,2)

    def test_ambiguous_initial_and_nearby_second_subject_pause_output(self):
        tracker=TrackedObjectDetector(SequenceDetector([[obj(),obj(300)],[obj()],[obj(140),obj(240)]]))
        self.assertFalse(tracker.detect_at(self.frame,0.).observations)
        tracker.detect_at(self.frame,.05)
        r=tracker.detect_at(self.frame,.1)
        self.assertFalse(r.observations)
        self.assertEqual(r.diagnostics['tracking']['state'],'ambiguous')

    def test_far_subject_cannot_steal_lock_and_timeout_starts_new_identity(self):
        tracker=TrackedObjectDetector(SequenceDetector([[obj(60)],[obj(490)],[obj(490)]]))
        tracker.detect_at(self.frame,0.)
        self.assertFalse(tracker.detect_at(self.frame,.05).observations)
        r=tracker.detect_at(self.frame,.6)
        self.assertEqual(r.observations[0].center,(490,120))
        self.assertEqual(r.diagnostics['tracking']['identity'],2)

    def test_missing_image_evidence_never_holds_a_fake_observation(self):
        tracker=TrackedObjectDetector(SequenceDetector([[obj()],[],[],[]]))
        tracker.detect_at(self.frame,0.)
        for now in [.05,.1,.4]:self.assertFalse(tracker.detect_at(self.frame,now).observations)

    def test_validated_flow_tracks_translation_without_a_model_confidence(self):
        rng=np.random.default_rng(17)
        pattern=rng.integers(0,256,(80,90,3),dtype=np.uint8)
        frames=[]
        for dx in [0,8,16]:
            f=self.frame.copy();f[80:160,105+dx:195+dx]=pattern;frames.append(f)
        detector=SequenceDetector([[obj()],[],[]]);tracker=TrackedObjectDetector(detector)
        tracker.detect_at(frames[0],0.)
        result=tracker.detect_at(frames[1],.04)
        self.assertEqual(result.diagnostics['tracking']['state'],'flow')
        self.assertLess(abs(result.observations[0].center[0]-158),3)
        self.assertIsNone(result.observations[0].score)
        # The timeout is measured from model confirmation, never renewed by flow.
        self.assertFalse(tracker.detect_at(frames[2],.2).observations)
        self.assertEqual(detector.calls,3)

    def test_flow_disappearance_and_geometric_failure_stop_instead_of_drifting(self):
        rng=np.random.default_rng(18);f=self.frame.copy();f[80:160,105:195]=rng.integers(0,256,(80,90,3),dtype=np.uint8)
        tracker=TrackedObjectDetector(SequenceDetector([[obj()],[],[]]))
        tracker.detect_at(f,0.)
        self.assertFalse(tracker.detect_at(self.frame,.04).observations)
        with patch.object(tracker,'_flow',side_effect=AssertionError('Failed flow must not restart')):
            self.assertFalse(tracker.detect_at(self.frame,.08).observations)

    def test_resolution_or_time_reset_cannot_flow_across_unrelated_frames(self):
        tracker=TrackedObjectDetector(SequenceDetector([[obj()],[],[obj()],[]]))
        tracker.detect_at(self.frame,1.)
        self.assertFalse(tracker.detect_at(np.zeros((600,1000,3),np.uint8),1.05).observations)
        tracker.detect_at(self.frame,2.)
        self.assertFalse(tracker.detect_at(self.frame,1.9).observations)

    def test_flow_cannot_extend_identity_indefinitely_at_high_frame_rate(self):
        tracker=TrackedObjectDetector(SequenceDetector([[obj()],[],[],[],[]]))
        tracker.detect_at(self.frame,0.)
        with patch.object(tracker,'_flow',return_value=obj()) as flow:
            for now in [.02,.04,.06]:
                self.assertEqual(tracker.detect_at(self.frame,now).diagnostics['tracking']['state'],'flow')
            self.assertFalse(tracker.detect_at(self.frame,.08).observations)
            self.assertEqual(flow.call_count,3)

    def test_live_preview_preserves_tracker_identity_and_source_coordinates(self):
        from visuallab.camera import run_preview
        from visuallab.terminal_status import PipelineInfo,TerminalDashboard
        frame=self.frame;original=frame.copy();callbacks=[]
        class Source:
            released=False
            def read(self):return True,frame
            def release(self):self.released=True
        source=Source()
        tracker=TrackedObjectDetector(SequenceDetector([[obj()]]))
        with patch('visuallab.camera.cv2.namedWindow'),patch('visuallab.camera.cv2.resizeWindow'), \
             patch('visuallab.camera.cv2.moveWindow'),patch('visuallab.camera.cv2.imshow'), \
             patch('visuallab.camera.cv2.destroyAllWindows'),patch('visuallab.camera.cv2.waitKey',return_value=ord('q')):
            run_preview(source=source,detector=tracker,trail_labels=frozenset({'armor'}),
                status=TerminalDashboard(PipelineInfo('armor'),mode='off'),
                on_result=lambda r,size:callbacks.append((r.observations[0].center,r.diagnostics['tracking'],size)))
        self.assertTrue(source.released)
        self.assertEqual(callbacks[0][0],(150,120))
        self.assertEqual(callbacks[0][1]['state'],'detector')
        self.assertEqual(callbacks[0][2],(500,300))
        np.testing.assert_array_equal(frame,original)

    def test_tracking_does_not_modify_source_and_binary_is_local(self):
        tracker=TrackedObjectDetector(SequenceDetector([[obj()]]));original=self.frame.copy()
        r=tracker.detect_at(self.frame,0.)
        np.testing.assert_array_equal(self.frame,original)
        self.assertFalse(r.binary[:60].any())
        self.assertEqual(r.gray.shape,self.frame.shape[:2])

    def test_associated_trail_uses_tracker_measurements_without_second_gate(self):
        effect=TrajectoryTrail(labels=frozenset({'armor'}))
        effect.update([obj(80)],self.frame.shape,0.,associated=True)
        effect.update([obj(320)],self.frame.shape,.05,associated=True)
        self.assertEqual(effect.points[-1].center,(320,120))
        self.assertFalse(effect.points[-1].start)
        before=self.frame.copy();effect.draw(self.frame,.1)
        self.assertFalse(np.array_equal(self.frame,before))
        # Particles and the thick line leave detector results unchanged.
        self.assertEqual(effect.target.center,(320,120))
        effect.reset();self.assertFalse(effect.points)


if __name__=='__main__':unittest.main()
