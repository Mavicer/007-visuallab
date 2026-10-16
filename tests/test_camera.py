"""Exercise the live loop without touching the real camera or opening windows."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from visuallab.camera import run_preview
from visuallab.detection import DetectionResult, Observation
from visuallab.preview_layout import PREVIEW_WINDOW, INFO_HEIGHT, compose_preview, preview_size, display_bounds


class CameraLoopTests(unittest.TestCase):
    def test_custom_frame_source_and_detector_use_same_preview(self) -> None:
        frame = np.full((240, 320, 3), 150, dtype=np.uint8)

        class Source:
            released = False

            def read(self):
                return True, frame

            def release(self):
                self.released = True

        class Detector:
            called = False

            def detect(self, image):
                self.called = True
                self_shape = image.shape
                assert self_shape == frame.shape
                return DetectionResult([Observation((80, 70, 40, 80), (100, 110), "subject", 0.8)])

        source, detector = Source(), Detector()
        delivered = []
        with (
            patch("visuallab.camera.cv2.VideoCapture") as make_capture,
            patch("visuallab.camera.cv2.namedWindow"),
            patch("visuallab.camera.cv2.resizeWindow"),
            patch("visuallab.camera.cv2.moveWindow"),
            patch("visuallab.camera.cv2.imshow") as show,
            patch("visuallab.camera.cv2.waitKey", return_value=ord("q")),
            patch("visuallab.camera.cv2.destroyAllWindows"),
        ):
            run_preview(
                source=source, detector=detector,
                on_result=lambda result, size: delivered.append((result, size)),
            )
        self.assertTrue(source.released)
        self.assertTrue(detector.called)
        make_capture.assert_not_called()
        self.assertEqual(show.call_count, 1)
        self.assertEqual(show.call_args.args[0],PREVIEW_WINDOW)
        self.assertEqual(delivered[0][1], (320, 240))
        self.assertEqual(delivered[0][0].observations[0].center, (100, 110))

    def test_processed_frame_is_shown_and_camera_released(self) -> None:
        frame = np.full((300, 400, 3), 235, dtype=np.uint8)
        cv2.putText(frame, "1", (140, 230), cv2.FONT_HERSHEY_SIMPLEX, 4,
                    (10, 10, 10), 9, cv2.LINE_AA)
        with (
            patch("visuallab.camera.cv2.VideoCapture") as make_capture,
            patch("visuallab.camera.cv2.namedWindow") as make_window,
            patch("visuallab.camera.cv2.resizeWindow") as resize_window,
            patch("visuallab.camera.cv2.moveWindow") as move_window,
            patch("visuallab.camera.cv2.imshow") as show,
            patch("visuallab.camera.cv2.waitKey", return_value=ord("q")),
            patch("visuallab.camera.cv2.destroyAllWindows") as close_windows,
            patch("visuallab.camera.display_bounds",return_value=(1800,960)),
        ):
            capture = make_capture.return_value
            capture.isOpened.return_value = True
            capture.read.return_value = (True, frame)
            run_preview()
            self.assertEqual(show.call_count, 1)
            self.assertEqual(
                {call.args[0] for call in show.call_args_list},
                {PREVIEW_WINDOW},
            )
            self.assertEqual(make_window.call_count, 1)
            sizes = {call.args[0]: call.args[1:] for call in resize_window.call_args_list}
            positions = {call.args[0]: call.args[1:] for call in move_window.call_args_list}
            self.assertEqual(set(sizes), set(positions))
            self.assertEqual(positions[PREVIEW_WINDOW],(20,50))
            self.assertEqual(sizes[PREVIEW_WINDOW],preview_size(frame.shape))
            displayed=show.call_args.args[1]
            self.assertEqual(displayed.shape[1::-1],sizes[PREVIEW_WINDOW])
            capture.release.assert_called_once()
            close_windows.assert_called_once()

    def test_compound_layout_keeps_color_gray_and_binary_in_fixed_cells(self):
        marked=np.full((300,400,3),(20,100,230),np.uint8)
        gray=np.full((300,400),77,np.uint8)
        binary=np.full((300,400),255,np.uint8)
        before=marked.copy()
        result=compose_preview(gray,binary,marked,info_lines=['FPS: 20 | Inference: 50ms'])
        height,width=result.shape[:2];sw,sh=width//3,height//2
        # Main cell width equals the sum of the two small-cell widths.
        main_width=2*sw
        self.assertEqual(main_width,2*(width-main_width))
        np.testing.assert_array_equal(result[sh,sw],(20,100,230))
        np.testing.assert_array_equal(result[sh//2,sw*2+sw//2],(77,77,77))
        np.testing.assert_array_equal(result[sh+sh//2,sw*2+sw//2],(255,255,255))
        np.testing.assert_array_equal(marked,before)

    def test_preview_bounds_and_exact_half_sizes_for_various_camera_aspects(self):
        for shape in ((1080,1920,3),(720,1280,3),(301,401,3),(1280,720,3)):
            width,height=preview_size(shape)
            self.assertLessEqual(width,1800)
            self.assertLessEqual(height,960)
            self.assertEqual(width%3,0)
            self.assertEqual(height%2,0)

    def test_information_band_never_covers_color_image(self):
        image=np.full((360,640,3),(40,70,90),np.uint8)
        image[:20,:20]=(0,0,255)
        image[-20:,-20:]=(0,255,0)
        gray=np.full((360,640),80,np.uint8)
        blank=compose_preview(gray,gray,image)
        hud=compose_preview(gray,gray,image,info_lines=['status']*5)
        main_width=hud.shape[1]*2//3
        np.testing.assert_array_equal(blank[INFO_HEIGHT:,:main_width],hud[INFO_HEIGHT:,:main_width])
        # Both the first and last source rows remain inside the image area.
        np.testing.assert_array_equal(hud[INFO_HEIGHT+5,5],(0,0,255))
        np.testing.assert_array_equal(hud[-6,main_width-6],(0,255,0))

    def test_retina_desktop_bounds_and_headless_fallback(self):
        size=SimpleNamespace(width=1470,height=956)
        def main_id():return 1
        def bounds(_):return SimpleNamespace(size=size)
        library=SimpleNamespace(CGMainDisplayID=main_id,CGDisplayBounds=bounds)
        with patch('visuallab.preview_layout.platform.system',return_value='Darwin'),patch(
                'visuallab.preview_layout.ctypes.CDLL',return_value=library):
            self.assertEqual(display_bounds(),(1390,826))
            size.width,size.height=2560,1440
            self.assertEqual(display_bounds(),(1800,960))
            size.width,size.height=0,0
            self.assertEqual(display_bounds(),(1440,800))

    def test_read_failure_still_releases_camera(self) -> None:
        with (
            patch("visuallab.camera.cv2.VideoCapture") as make_capture,
            patch("visuallab.camera.cv2.destroyAllWindows") as close_windows,
        ):
            capture = make_capture.return_value
            capture.isOpened.return_value = True
            capture.read.return_value = (False, None)
            with self.assertRaisesRegex(RuntimeError, "读取画面失败"):
                run_preview()
            capture.release.assert_called_once()
            close_windows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
