"""Dashboard boundary checks: fake clocks, resources, models and camera."""
from io import StringIO
from types import SimpleNamespace
from pathlib import Path
import os
import plistlib
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from visuallab.terminal_status import (
    PipelineInfo, Resources, ResourceSampler, TerminalDashboard, apple_gpu_percent,
)
from visuallab.detection import DetectionResult, Observation
from visuallab.camera import run_preview
from visuallab.__main__ import _pipeline_info


class Sampler:
    snapshot = Resources(25.,180.,512*1024**2,10*1024**3,16*1024**3,0.,None)
    started = False
    closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


class Tty(StringIO):
    def isatty(self):
        return True


class DashboardTests(unittest.TestCase):
    frame = np.zeros((240,320,3),np.uint8)

    def test_gpu_zero_missing_invalid_and_plain_registry_output(self):
        self.assertEqual(apple_gpu_percent(plistlib.dumps([
            {"PerformanceStatistics":{"Device Utilization %":0}}])),0.)
        self.assertEqual(apple_gpu_percent(b'| "PerformanceStatistics" = {"Device Utilization %"=37,"other"=9}'),37.)
        for data in (b'broken',b'<?xml version="1.0"?><plist><broken',
                     b'"PerformanceStatistics" = {"Device Utilization %"=110}',
                     plistlib.dumps([{"PerformanceStatistics":{"Device Utilization %":False}}])):
            self.assertIsNone(apple_gpu_percent(data))

    def test_gpu_command_timeout_is_unavailable(self):
        with patch('visuallab.terminal_status.platform.system',return_value='Darwin'), patch(
            'visuallab.terminal_status.subprocess.run',side_effect=subprocess.TimeoutExpired('ioreg',.4)) as run:
            self.assertIsNone(ResourceSampler()._gpu())
        self.assertEqual(run.call_args.kwargs['timeout'],.4)

    def test_plain_refresh_is_limited_and_stale_text_is_cleared(self):
        stream,sampler = StringIO(),Sampler()
        now=[0.]
        dashboard=TerminalDashboard(PipelineInfo('text',('text-pilot-v1.pt | CPU',)),
            stream=stream,sampler=sampler,clock=lambda:now[0])
        dashboard.start()
        result=DetectionResult([Observation((1,2,3,4),(2,4),'A',.95)],score_name='ocr')
        dashboard.update(self.frame,result,12.)
        first=stream.getvalue()
        self.assertIn('text-pilot-v1.pt',first)
        self.assertIn('180.0%',first)  # Process CPU can exceed one core.
        self.assertIn('GPU 全机活动: 0.0%',first)
        now[0]=.1;dashboard.update(self.frame,result,12.)
        self.assertEqual(first,stream.getvalue())
        now[0]=.6;dashboard.update(self.frame,DetectionResult([]),10.)
        last=stream.getvalue()[len(first):]
        self.assertIn('[]',last)
        self.assertNotIn('"A"',last)
        self.assertNotIn('\x1b',stream.getvalue())
        dashboard.close()
        self.assertTrue(sampler.started and sampler.closed)

    def test_tty_sanitizes_ocr_and_restores_cursor(self):
        stream,sampler=Tty(),Sampler()
        dashboard=TerminalDashboard(PipelineInfo('text'),stream=stream,sampler=sampler)
        result=DetectionResult([Observation((1,2,3,4),(2,4),'evil\x1b[31m\nnext',.99),
            Observation((10,20,3,4),(11,22),'?',.95),
            Observation((20,20,3,4),(21,22),'?',.3,is_unknown=True)],score_name='ocr')
        with patch.dict(os.environ,{'TERM':'xterm-256color'}),patch(
            'visuallab.terminal_status.shutil.get_terminal_size',return_value=os.terminal_size((100,24))):
            dashboard.start();dashboard.update(self.frame,result,12.);dashboard.close()
        value=stream.getvalue()
        self.assertIn('\x1b[?25l',value)
        self.assertIn('\x1b[?25h',value)
        self.assertIn('\x1b[?1049h',value)
        self.assertIn('\x1b[?1049l',value)
        self.assertNotIn('\x1b[31m',value)
        self.assertIn('OK "?"',value)
        self.assertIn('UNKNOWN 未知',value)

    def test_closed_output_does_not_prevent_resource_cleanup(self):
        stream=StringIO();stream.close();sampler=Sampler()
        panel=TerminalDashboard(PipelineInfo('text'),stream=stream,sampler=sampler)
        panel.start();panel.update(self.frame,DetectionResult([]));panel.close()
        self.assertTrue(sampler.closed)

    def test_narrow_terminal_restores_screen_and_uses_plain_logs(self):
        stream,sampler=Tty(),Sampler();now=[0.]
        panel=TerminalDashboard(PipelineInfo('text'),stream=stream,sampler=sampler,clock=lambda:now[0])
        with patch.dict(os.environ,{'TERM':'xterm'}),patch(
            'visuallab.terminal_status.shutil.get_terminal_size',return_value=os.terminal_size((100,24))):
            panel.start();panel.update(self.frame,DetectionResult([]))
        now[0]=1.
        with patch('visuallab.terminal_status.shutil.get_terminal_size',return_value=os.terminal_size((40,12))):
            panel.update(self.frame,DetectionResult([]));panel.close()
        self.assertIn('\x1b[?1049l',stream.getvalue())
        self.assertTrue(sampler.closed)

    def test_off_does_not_start_sampling_but_keeps_save_notifications(self):
        stream,sampler=StringIO(),Sampler()
        panel=TerminalDashboard(PipelineInfo('raw'),mode='off',stream=stream,sampler=sampler)
        panel.start();panel.update(self.frame,None);panel.notify('已保存：frame.png');panel.close()
        self.assertFalse(sampler.started)
        self.assertIn('frame.png',stream.getvalue())
        self.assertNotIn('GPU',stream.getvalue())

    def test_camera_failure_closes_panel_and_releases_source(self):
        source=SimpleNamespace(read=lambda:(False,None),release=lambda:releases.append(True))
        releases=[];sampler=Sampler()
        panel=TerminalDashboard(PipelineInfo('raw'),stream=StringIO(),sampler=sampler)
        with patch('visuallab.camera.cv2.destroyAllWindows'):
            with self.assertRaisesRegex(RuntimeError,'读取画面失败'):
                run_preview(source=source,mode='raw',status=panel)
        self.assertEqual(releases,[True])
        self.assertTrue(sampler.started and sampler.closed)

    def test_hybrid_models_show_mps_detector_and_cpu_ocr_separately(self):
        args=SimpleNamespace(backend='hybrid',text_locator='db',mode='adaptive',reader='ocr',
            no_recognition=False,weights=Path('card.pt'),device='mps',yolo_size=640,
            text_layout='detect',text_model=None,ocr_model=Path('ocr.onnx'),target_roi=None,camera=0)
        info=_pipeline_info(args)
        self.assertIn('PyTorch/MPS',info.models[0])
        self.assertIn('ONNX/CPU',info.models[1])


if __name__=='__main__':
    unittest.main()
