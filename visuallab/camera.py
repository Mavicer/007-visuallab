"""Camera input and live views for the current learning stages."""

from datetime import datetime
from pathlib import Path
from typing import Callable
from time import perf_counter

import cv2
import numpy as np

from .detection import DetectionResult, FrameDetector, FrameSource, NumberCardDetector
from .terminal_status import PipelineInfo, TerminalDashboard
from .preview_layout import PREVIEW_WINDOW, compose_preview, preview_size, display_bounds
from .trajectory import TrajectoryTrail


def _preview_size(shape: tuple[int, ...], max_width: int, max_height: int) -> tuple[int, int]:
    """Fit a camera frame inside a display window without changing processing."""
    height, width = shape[:2]
    scale = min(max_width / width, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _arrange_windows(shape: tuple[int, ...], mode: str,
                     limits: tuple[int,int] = (1800,960)) -> None:
    """Create one native window; child views cannot overlap or reorder."""
    if mode == "raw":
        name,size = "VisualLab - Camera Preview",_preview_size(shape,760,420)
    else:
        name,size = PREVIEW_WINDOW,preview_size(shape,*limits)
    cv2.namedWindow(name,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name,*size)
    cv2.moveWindow(name,20,50)


def render_result(
    frame: np.ndarray, result: DetectionResult, mode: str, *, summary: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build source-resolution views for the unified preview and offline checks."""
    marked = frame.copy()
    cv2.drawContours(marked, result.regions, -1, (255, 255, 0), 2)
    for observation in result.observations:
        x, y, width, height = observation.box
        label, score = observation.label, observation.score
        color = (0, 180, 255) if observation.is_unknown else (0, 255, 0)
        if observation.polygon is None:
            cv2.rectangle(marked, (x, y), (x + width, y + height), color, 2)
        else:
            cv2.polylines(marked, [observation.polygon], True, color, 2)
        if label is not None:
            title = (
                f"{label} {result.score_name}:{score:.2f}"
                if not observation.is_unknown and score is not None else label
            )
            cv2.putText(marked, title, (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if summary:
        cv2.putText(marked,
            f"{result.region_name}: {len(result.regions)} | Subjects: {len(result.observations)} | {mode}",
            (12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
    gray = result.gray if result.gray is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = result.binary if result.binary is not None else np.zeros_like(gray)
    return gray, binary, marked


def run_preview(
    camera_index: int = 0,
    mode: str = "adaptive",
    min_area: int = 120,
    recognize: bool = True,
    source: FrameSource | None = None,
    detector: FrameDetector | None = None,
    on_result: Callable[[DetectionResult, tuple[int, int]], None] | None = None,
    status: TerminalDashboard | None = None,
    preview_width: int = 1800,
    trail: bool = True,
    trail_labels: frozenset[str] = frozenset({'+'}),
) -> None:
    """Display live frames until the user presses Q or Escape."""
    if camera_index < 0:
        raise ValueError("摄像头编号不能为负数")

    # macOS 的 AVFoundation 后端负责连接本机摄像头。
    capture = source if source is not None else cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if hasattr(capture, "isOpened") and not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"无法打开摄像头 {camera_index}。请确认摄像头未被占用，并在 macOS 系统设置中允许终端或 VS Code 使用摄像头。"
        )

    # 检测器只创建一次；后续追踪器或其他主体检测器可复用同一预览循环。
    active_detector = detector if detector is not None else (
        NumberCardDetector(mode, min_area, recognize) if mode != "raw" else None
    )
    windows_ready = False
    trajectory = TrajectoryTrail(enabled=trail,labels=trail_labels)
    trail_identity = None
    limits = display_bounds(preview_width,960)
    panel = status if status is not None else TerminalDashboard(PipelineInfo(
        type(active_detector).__name__ if active_detector is not None else "原始预览",
        scope=f"摄像头 {camera_index}"))
    print("摄像头已打开。S 保存原始帧；Q/Esc 退出。" if mode == "raw" else
          "摄像头已打开。S 保存原始帧；T 开关加号拖尾；C 清除拖尾；Q/Esc 退出。")
    try:
        panel.start()
        while True:
            # read() 每次返回是否成功和一帧图像；处理函数不负责摄像头资源。
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("摄像头已打开，但读取画面失败。请检查摄像头连接或权限。")

            if not windows_ready:
                # 取得实际帧尺寸后再布置窗口；识别仍用原分辨率。
                _arrange_windows(frame.shape, mode,limits)
                windows_ready = True

            result = None
            inference_ms = 0.
            if mode == "raw":
                cv2.imshow("VisualLab - Camera Preview", frame)
            else:
                assert active_detector is not None
                started = perf_counter()
                result = active_detector.detect(frame)
                inference_ms = (perf_counter()-started)*1000
                # Stage 4 can attach a tracker or controller without making
                # this display loop depend on any particular motor hardware.
                if on_result is not None:
                    on_result(result, (frame.shape[1], frame.shape[0]))
                gray, binary, marked = render_result(frame, result, mode,summary=False)
                now = perf_counter()
                tracking = result.diagnostics.get('tracking')
                if tracking is not None and tracking['identity'] != trail_identity:
                    trajectory.reset()  # A new lock must never bridge old trails.
                    trail_identity = tracking['identity']
                trajectory.update(result.observations,frame.shape,now,
                                  associated=tracking is not None)
                trajectory.draw(marked,now)
                info_lines = list(panel.preview_lines(result,inference_ms))
                if info_lines:
                    state = "off" if not trajectory.enabled else "active" if trajectory.visible else "waiting"
                    info_lines[0] += f" | Trail: {state}"
                    if tracking is not None:
                        info_lines[0] += f" | Track: {tracking['state']} #{tracking['identity']}"
                workspace=compose_preview(gray,binary,marked,
                    info_lines=info_lines,
                    max_width=limits[0],max_height=limits[1])
                cv2.imshow(PREVIEW_WINDOW,workspace)
            # waitKey 同时处理窗口事件；按键判断需要预览窗口处于前台。
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                # 仅在用户主动按键时保存无框原图，便于复现轮廓粘连。
                captures = Path(__file__).resolve().parents[1] / "captures"
                captures.mkdir(exist_ok=True)
                target = captures / f"frame-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
                if cv2.imwrite(str(target), frame):
                    panel.notify(f"已保存原始帧：{target}")
                else:
                    panel.notify(f"原始帧保存失败：{target}")
            elif mode != "raw" and key in (ord("t"),ord("T")):
                trajectory.enabled = not trajectory.enabled
                trajectory.reset()
                panel.notify("加号拖尾已开启" if trajectory.enabled else "加号拖尾已关闭")
            elif mode != "raw" and key in (ord("c"),ord("C")):
                trajectory.reset()
                panel.notify("已清除加号拖尾")
            # The sampler runs independently; frame updates only read its
            # latest snapshot and redraw at a bounded terminal frequency.
            panel.update(frame, result, inference_ms)
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        try:
            capture.release()
        finally:
            try:
                cv2.destroyAllWindows()
            finally:
                panel.close()
