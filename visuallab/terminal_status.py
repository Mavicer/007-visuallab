"""Small terminal dashboard; sampling never runs in the inference loop.

GPU counters are system-wide driver snapshots, not this process's GPU share.
Unavailable metrics stay None instead of becoming a misleading zero.
"""
from collections import deque
from dataclasses import dataclass
import json
import math
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import threading
from time import monotonic
import unicodedata
from xml.parsers.expat import ExpatError

import psutil


@dataclass(frozen=True)
class PipelineInfo:
    backend: str
    models: tuple[str, ...] = ()
    scope: str = "全画面"


@dataclass(frozen=True)
class Resources:
    system_cpu: float | None = None
    process_cpu: float | None = None  # 100% means one fully occupied core.
    rss: int | None = None
    memory_used: int | None = None
    memory_total: int | None = None
    gpu: float | None = None
    mps_memory: int | None = None


def apple_gpu_percent(payload: bytes) -> float | None:
    """Read a public driver counter when present; tolerate OS/driver changes."""
    values = []
    try:
        entries = plistlib.loads(payload)
        for entry in entries:
            value = entry.get("PerformanceStatistics", {}).get("Device Utilization %")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 100:
                values.append(float(value))
    except (ValueError, TypeError, AttributeError, plistlib.InvalidFileException, ExpatError):
        # Some macOS environments refuse ioreg archive output while ordinary
        # read-only text output works. Parse only the exact driver metric.
        for line in payload.decode("utf-8",errors="replace").splitlines():
            if '"PerformanceStatistics"' in line:
                for match in re.findall(r'"Device Utilization %"\s*=\s*([0-9]+(?:\.[0-9]+)?)', line):
                    value = float(match)
                    if 0 <= value <= 100:
                        values.append(value)
    return max(values) if values else None


class ResourceSampler:
    """One bounded daemon worker, one sample/sec, no sudo or model imports."""

    def __init__(self) -> None:
        self.snapshot = Resources()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="visuallab-resources", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _gpu(self) -> float | None:
        if platform.system() != "Darwin":
            return None
        try:
            result = subprocess.run(["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "AGXAccelerator", "-l", "-w", "0"],
                                    capture_output=True, timeout=.4, check=True)
            return apple_gpu_percent(result.stdout)
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _mps_memory() -> int | None:
        # Reuse torch only if the chosen detector already imported it. This
        # must not make the plain OpenCV path load an entire DL framework.
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.backends.mps.is_available():
                return int(torch.mps.current_allocated_memory())
        except (AttributeError, RuntimeError):
            pass
        return None

    def _run(self) -> None:
        try:
            process = psutil.Process()
            psutil.cpu_percent(interval=None)
            process.cpu_percent(interval=None)
        except (psutil.Error, OSError):
            return
        first = True
        while not self._stop.is_set():
            try:
                memory = psutil.virtual_memory()
                # The first nonblocking CPU reading has no time interval;
                # show N/A until the next sample rather than displaying 0%.
                cpu = None if first else psutil.cpu_percent(interval=None)
                app_cpu = None if first else process.cpu_percent(interval=None)
                self.snapshot = Resources(cpu, app_cpu, process.memory_info().rss,
                    memory.total-memory.available, memory.total, self._gpu(), self._mps_memory())
            except (psutil.Error, OSError):
                self.snapshot = Resources()
            first = False
            if self._stop.wait(1.):
                break

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=.6)


def _clean(value: object) -> str:
    """OCR/model text must never inject terminal controls or extra rows."""
    return "".join(c if c.isprintable() else f"\\x{ord(c):02x}" for c in str(value))


def _fit(text: str, width: int) -> str:
    used, output = 0, []
    for character in _clean(text):
        size = 0 if unicodedata.combining(character) else 2 if unicodedata.east_asian_width(character) in "WF" else 1
        if used + size > width:
            return "".join(output[:-1]) + "…" if output else ""
        output.append(character)
        used += size
    return "".join(output)


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


def _memory(value: int | None) -> str:
    return "N/A" if value is None else f"{value/1024**2:.0f} MiB"


class TerminalDashboard:
    """ANSI redraw on a TTY; bounded plain logs for pipes/IDE consoles."""

    def __init__(self, info: PipelineInfo, *, mode: str = "auto", interval: float = .5,
                 stream=None, sampler=None, clock=monotonic) -> None:
        self.info, self.mode, self.interval = info, mode, interval
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock
        self.sampler = sampler if sampler is not None else ResourceSampler()
        self._last = -math.inf
        self._screen_active = False
        self._plain_header = False
        self._started = False
        self._closed = False
        self._disabled = mode == "off"
        self._history = deque(maxlen=120)
        self.frames = 0
        self.event = "就绪；Q/Esc 退出，S 保存原始帧（画面窗口需在前台）"

    def start(self) -> None:
        if not self._disabled and not self._started:
            self.sampler.start()
            self._started = True

    def notify(self, message: str) -> None:
        self.event = _clean(message)
        if self.mode == "off":
            try:
                self.stream.write(self.event+"\n")
                self.stream.flush()
            except (OSError,ValueError):
                pass

    @property
    def loop_fps(self) -> float | None:
        if len(self._history)>1 and self._history[-1]>self._history[0]:
            return (len(self._history)-1)/(self._history[-1]-self._history[0])
        return None

    def preview_lines(self, result, inference_ms: float) -> list[str]:
        """Read cached metrics for the image HUD; never sample on this path."""
        fps = "--" if self.loop_fps is None else f"{self.loop_fps:.1f}"
        backend = result.diagnostics.get("mode",self.info.backend)
        accepted = sum(item.label is not None and not item.is_unknown for item in result.observations)
        lines = [f"Backend: {backend} | Subjects: {len(result.observations)} | Accepted: {accepted}",
                 f"Loop FPS: {fps} | Inference: {inference_ms:.1f} ms"]
        for model in self.info.models[:2]:
            parts=model.split(" | ")
            name=parts[0].split(" ",1)[-1]
            device=parts[1] if len(parts)>1 else ""
            lines.append(f"Model: {name} | {device}")
        resources=self.sampler.snapshot
        lines.append(f"CPU(sys): {_percent(resources.system_cpu)} | GPU(sys): {_percent(resources.gpu)}")
        return lines

    def _lines(self, frame, result, inference_ms: float, now: float) -> list[str]:
        size = shutil.get_terminal_size((100, 30))
        observations = result.observations if result is not None else []
        accepted = sum(o.label is not None and not o.is_unknown for o in observations)
        unknown = sum(o.is_unknown for o in observations)
        fps = self.loop_fps
        resources = self.sampler.snapshot
        fps_text = "采样中" if fps is None else f"{fps:.1f}"
        memory = ("N/A" if resources.memory_used is None or not resources.memory_total else
                  f"{resources.memory_used/1024**3:.1f}/{resources.memory_total/1024**3:.1f} GiB")
        tracking = result.diagnostics.get('tracking') if result is not None else None
        title = "短时目标 ID" if tracking is not None else "本帧结果，无跨帧 ID"
        lines = [f"VisualLab | 实时终端状态（{title}）",
                 f"后端: {self.info.backend} | 范围: {self.info.scope}"]
        lines.extend(f"模型: {model}" for model in self.info.models)
        if tracking is not None:
            lines.append(f"追踪: {tracking['state']} | ID: {tracking['identity']} | 检出间隔: {tracking['detection_age_ms']} ms（flow 无模型分数）")
        lines.extend([
            f"画面: {frame.shape[1]}x{frame.shape[0]} | 帧: {self.frames} | 实际循环 FPS: {fps_text} | 推理: {inference_ms:.1f} ms",
            f"CPU 全机: {_percent(resources.system_cpu)} | 本进程: {_percent(resources.process_cpu)}（单核满载=100%）",
            f"内存 本进程 RSS: {_memory(resources.rss)} | 全机使用: {memory}",
            f"GPU 全机活动: {_percent(resources.gpu)}（驱动快照） | 本进程 MPS 张量: {_memory(resources.mps_memory)}",
            f"本帧候选: {len(observations)} | 有输出: {accepted} | 未知: {unknown}",
        ])
        timings = result.diagnostics.get("timings_ms", {}) if result is not None else {}
        stages = [(label,timings[key]) for key,label in (("object_localization","对象"),("text_localization","文字定位"),("ocr","OCR"),("recovery","姿态恢复"))
                  if isinstance(timings.get(key),(float,int))]
        if stages:
            lines.append("分阶段: " + " | ".join(f"{label} {value:.1f} ms" for label,value in stages))
        lines.append("行号 / 状态 / 识别结果 / 分数 / 定位分数 / 中心 / 框 xywh")
        # Fit the terminal height; never let a noisy frame produce an
        # unbounded scrolling table. Remaining count is always visible.
        room = min(8, max(1, size.lines-len(lines)-4))
        for index, item in enumerate(observations[:room],1):
            label = "未知" if item.is_unknown else "未读取" if item.label is None else _fit(json.dumps(item.label,ensure_ascii=False),24)
            score = "--" if item.score is None else f"{result.score_name}={item.score:.3f}"
            source = item.source_score
            details = result.diagnostics.get("inputs", [])
            if source is None and index <= len(details):
                source = details[index-1].get("text_detection_score")
            source_text = "--" if source is None else f"det={source:.3f}"
            state = "UNKNOWN" if item.is_unknown else "ROI" if item.label is None else "OK"
            if index <= len(details) and details[index-1].get("recovery",{}).get("status") == "recovered":
                state = "RECOVERED"
            lines.append(f"{index:02d} {state} {label} | {score} | {source_text} | {item.center} | {item.box}")
        if not observations:
            lines.append("  无文字/目标候选" if result is not None else "  原始预览：未启用识别")
        if len(observations)>room:
            lines.append(f"  另有 {len(observations)-room} 个候选未展开")
        lines.extend([self.event,"GPU 非进程占比；N/A=未取得统计；分数不等于正确概率。"])
        return [_fit(line,max(1,size.columns-1)) for line in lines]

    def update(self, frame, result, inference_ms: float = 0., *, force: bool = False) -> None:
        if self._closed:
            return
        now = self.clock()
        self.frames += 1
        self._history.append(now)
        while len(self._history)>2 and now-self._history[1]>2:
            self._history.popleft()
        if self._disabled:
            return
        if not force and now-self._last < self.interval:
            return
        self._last = now
        lines = self._lines(frame,result,inference_ms,now)
        size = shutil.get_terminal_size((100,30))
        try:
            tty = (self.mode == "auto" and self.stream.isatty() and os.environ.get("TERM") != "dumb"
                   and size.columns >= 60 and size.lines >= 20)
            if tty:
                # An alternate screen keeps shell history intact. Redraw
                # from home so native library stderr cannot shift the panel.
                prefix = "" if self._screen_active else "\x1b[?1049h\x1b[?25l"
                self._screen_active = True
                self.stream.write(prefix + "\x1b[H\x1b[2J" + "\n".join(lines)+"\n")
            else:
                if self._screen_active:
                    self.stream.write("\x1b[?25h\x1b[?1049l")
                    self._screen_active = False
                # Non-TTY receives a one-time model header and compact
                # snapshots; no cursor escapes or full table flood.
                if not self._plain_header or force:
                    self.stream.write("\n".join(lines)+"\n")
                    self._plain_header = True
                else:
                    observations = result.observations if result is not None else []
                    texts = ["<unknown>" if item.is_unknown else item.label for item in observations[:8]]
                    resources = self.sampler.snapshot
                    self.stream.write(f"[VisualLab 帧{self.frames}] {_clean(json.dumps(texts,ensure_ascii=False))} | "
                        f"推理 {inference_ms:.1f}ms | CPU {_percent(resources.system_cpu)} / 进程 {_percent(resources.process_cpu)} | "
                        f"RSS {_memory(resources.rss)} | GPU {_percent(resources.gpu)}\n")
            self.stream.flush()
        except (OSError, ValueError):
            self._disabled = True  # A closed pipe must not kill camera cleanup.

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self.sampler.close()
        if self._screen_active:
            try:
                self.stream.write("\x1b[?25h\x1b[?1049l")
                self.stream.flush()
            except (OSError,ValueError):
                pass
