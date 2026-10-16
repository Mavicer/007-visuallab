"""Run the current VisualLab stage with ``python -m visuallab``."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import cv2

from .camera import render_result, run_preview
from .detection import FrameDetector, NumberCardDetector
from .text_ocr import CardTextDetector, DEFAULT_OCR_MODEL
from .terminal_status import PipelineInfo, TerminalDashboard
from .preview_layout import compose_preview


def _pipeline_info(args) -> PipelineInfo:
    """Describe the models actually enabled by this CLI invocation."""
    models = []
    yolo = args.backend in ("yolo", "hybrid", "armor") or (args.backend == "text" and args.text_locator == "yolo")
    if yolo:
        models.append(f"定位 {args.weights.name} | PyTorch/{args.device.upper()} | imgsz={args.yolo_size}")
    if args.mode != "raw" and (args.reader == "located-ocr" or
            (args.backend == "text" and args.text_layout == "detect" and args.text_locator == "db")):
        from .text_regions import DEFAULT_TEXT_MODEL
        models.append(f"文字定位 {(args.text_model or DEFAULT_TEXT_MODEL).name} | ONNX/CPU")
    if args.mode != "raw" and args.backend not in ("yolo", "armor") and not args.no_recognition:
        models.append("读取 1/2 模板 | OpenCV/CPU" if args.reader == "template" else
                      f"读取 {args.ocr_model.name} | ONNX/CPU")
    backend = "原始预览" if args.mode == "raw" else args.backend
    image_path = getattr(args, "image", None)
    scope = f"图片 {image_path.name}" if image_path is not None else f"摄像头 {args.camera}"
    if args.target_roi:
        scope += " | ROI " + " ".join(f"{value:g}" for value in args.target_roi)
    elif args.backend == "text":
        scope += " | 全画面文字"
    elif args.backend == "armor":
        scope += f" | {args.armor_class} 整板定位 + 短时光流（无 OCR）"
    if getattr(args,"text_recovery","off") == "auto":
        scope += " | 姿态恢复 auto（每帧最多两处）"
    if image_path is None and args.mode != "raw":
        scope += " | 目标拖尾 " + getattr(args,"trail","on")
    return PipelineInfo(backend, tuple(models), scope)


def _offline_image(image_path: Path, output_dir: Path, detector: FrameDetector, mode: str,
                   status: TerminalDashboard | None = None, preview_width: int = 1800) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"无法读取图片：{image_path}（请使用 PNG/JPEG）")
    try:
        if status is not None:
            status.start()
        start = perf_counter()
        result = detector.detect(frame)
        elapsed_ms = round((perf_counter() - start) * 1000, 1)
        if status is not None:
            status.notify("离线单帧；CPU 未完成采样时为 N/A，无视频 FPS")
            status.update(frame, result, elapsed_ms, force=True)
    finally:
        if status is not None:
            status.close()
    gray, binary, marked = render_result(frame, result, mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    _,_,clean_marked=render_result(frame,result,mode,summary=False)
    workspace = compose_preview(gray,binary,clean_marked,
        info_lines=status.preview_lines(result,elapsed_ms) if status is not None else (),
        max_width=preview_width)
    for name, image in (("gray.png", gray), ("binary.png", binary), ("result.png", marked),
                        ("workspace.png",workspace)):
        if not cv2.imwrite(str(output_dir / name), image):
            raise OSError(f"图片保存失败：{output_dir / name}")
    if result.debug_crops:
        crop_dir = output_dir / "ocr_inputs"
        crop_dir.mkdir(exist_ok=True)
        for name, crop in result.debug_crops.items():
            if not cv2.imwrite(str(crop_dir / f"{name}.png"), crop):
                raise OSError(f"OCR 裁剪保存失败：{name}")
    report = {
        "input": str(image_path), "processing_ms_single_frame": elapsed_ms,
        "score_name": result.score_name,
        "regions": len(result.regions),
        "diagnostics": result.diagnostics,
        "observations": [
            {"box_xywh": list(item.box), "center": list(item.center),
             "label": item.label, "score": item.score,
             "is_unknown": item.is_unknown,
             "card_detection_confidence": item.source_score}
            for item in result.observations
        ],
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已保存离线结果：{output_dir}（单帧 {elapsed_ms} ms，仅供本机参考）")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisualLab：摄像头或静态图片识别")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头编号，默认 0（通常是内建摄像头）",
    )
    parser.add_argument(
        "--mode",
        choices=("adaptive", "otsu", "raw"),
        default="adaptive",
        help="二值化方式；raw 保留第一阶段的原始画面（默认：adaptive）",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=120,
        help="只为像素数达到此值的纸面内笔画画框（默认：120）",
    )
    parser.add_argument(
        "--no-recognition",
        action="store_true",
        help="只看纸面和笔画候选，不显示数字标签",
    )
    parser.add_argument("--backend", choices=("traditional", "yolo", "hybrid", "text", "armor"),
                        default="traditional", help="检测后端（默认：traditional）")
    parser.add_argument("--reader", choices=("ocr", "template", "located-ocr"), default="ocr",
                        help="ocr 为通用字符读取；template 为旧 1/2 对照；located-ocr 为可选卡内文本定位")
    parser.add_argument("--ocr-model", type=Path, default=DEFAULT_OCR_MODEL,
                        help="本地 PP-OCRv4 英文 ONNX 模型路径")
    parser.add_argument("--text-model", type=Path, help="可选 DB 文字定位 ONNX；未指定时使用固定官方模型")
    parser.add_argument("--text-locator", choices=("db", "yolo"), default="db",
                        help="text 后端定位器；yolo 使用含 text 类的 --weights")
    parser.add_argument("--target-roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="text 后端的目标范围；归一化 x y w h，数值位于 0～1")
    parser.add_argument("--full-frame-text", action="store_true",
                        help="显式扫描全画面文字（诊断用途，背景文字也会被读取）")
    parser.add_argument("--text-layout", choices=("detect", "direct"), default="detect",
                        help="text 后端：detect 定位文本行；direct 直接读指定单字符/单行 ROI")
    parser.add_argument("--text-recovery", choices=("off", "auto"), default="off",
                        help="读不出时进行有预算的透视/方向恢复（默认 off；每帧最多两处）")
    parser.add_argument("--weights", type=Path, help="本地 YOLO .pt 权重；YOLO/hybrid 必填")
    parser.add_argument("--card-class", default="card", help="hybrid 使用的纸片类别名")
    parser.add_argument("--armor-class", default="armor", help="armor 后端跟踪的整板类别名")
    parser.add_argument("--armor-confidence", type=float, default=.5,
                        help="整板模型接受阈值，默认 .5（独立于通用文字模型）")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="YOLO 推理设备（默认：cpu；Apple Silicon 可试 mps）")
    parser.add_argument("--yolo-size", type=int, default=640,
                        help="YOLO 输入尺寸；本轮试验模型评估使用 416")
    parser.add_argument("--image", type=Path, help="离线处理一张 PNG/JPEG，省去摄像头")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/preview"),
                        help="离线输出目录（默认：runs/preview）")
    parser.add_argument("--terminal", choices=("auto", "log", "off"), default="auto",
                        help="终端状态：auto 原位刷新，log 文本快照，off 关闭面板")
    parser.add_argument("--status-interval", type=float, default=.5,
                        help="终端刷新间隔秒，至少 0.2；资源固定每秒后台采样")
    parser.add_argument("--preview-width",type=int,default=1800,
                        help="预览总宽度上限，默认 1800；实时窗口自动适配屏幕可用尺寸")
    parser.add_argument("--trail",choices=("on","off"),default="on",
                        help="实时彩色画面的加号轨迹拖尾，默认 on；T 开关、C 清除")
    args = parser.parse_args()
    if args.min_area <= 0:
        parser.error("--min-area 必须大于 0")
    if not 600 <= args.preview_width <= 3840:
        parser.error("--preview-width 必须在 600～3840 之间")
    if not math.isfinite(args.status_interval) or args.status_interval < .2:
        parser.error("--status-interval 必须为至少 0.2 的有限数值")
    if args.yolo_size <= 0:
        parser.error("--yolo-size 必须大于 0")
    if not 0 < args.armor_confidence < 1:
        parser.error("--armor-confidence 必须在 0 与 1 之间")
    if args.mode == "raw" and (args.image or args.backend != "traditional"):
        parser.error("raw 模式只用于摄像头原始预览")
    if args.backend == "text":
        if bool(args.target_roi) == args.full_frame_text:
            parser.error("text 后端需二选一：--target-roi X Y W H 或 --full-frame-text")
        if args.reader != "ocr":
            parser.error("text 后端使用 --reader ocr（默认）")
        if args.text_layout == "direct" and args.target_roi is None:
            parser.error("direct 只用于明确选定的单字符/单行 --target-roi")
    elif args.target_roi or args.full_frame_text or args.text_layout != "detect":
        parser.error("目标 ROI 和 text-layout 参数仅用于 --backend text")
    if args.reader == "located-ocr" and args.backend != "hybrid":
        parser.error("located-ocr 仅用于 hybrid；通用文字定位请用 --backend text")
    if args.text_recovery != "off" and not (args.backend == "text" or args.reader == "located-ocr"):
        parser.error("--text-recovery 需要 text 后端或 located-ocr 读取器")
    if args.text_model and not (args.backend == "text" or args.reader == "located-ocr"):
        parser.error("--text-model 需要 text 后端或 located-ocr 读取器")
    if args.text_locator == "yolo":
        if args.backend != "text" or args.text_layout == "direct":
            parser.error("--text-locator yolo 需要 text 后端和 detect 布局")
        if args.weights is None or args.text_model:
            parser.error("YOLO 文字定位需要 --weights；不同时使用 --text-model")
    detector: FrameDetector | None = None
    reader = None
    if args.backend not in ("yolo", "armor") and args.mode != "raw":
        try:
            if args.backend == "text" or args.reader == "located-ocr":
                from .text_regions import DBTextLocator, DEFAULT_TEXT_MODEL, RegionTextDetector
                direct = args.text_layout == "direct"
                if args.text_locator == "yolo":
                    from .yolo import YoloObjectDetector, YoloTextLocator
                    locator = YoloTextLocator(YoloObjectDetector(args.weights, device=args.device,
                                                                image_size=args.yolo_size),
                                              recover_empty=args.text_recovery == "auto")
                else:
                    locator = None if direct else DBTextLocator(args.text_model or DEFAULT_TEXT_MODEL)
                reader = RegionTextDetector(locator=locator, model_path=args.ocr_model,
                    direct=direct, target_roi=tuple(args.target_roi) if args.target_roi else None,
                    paper_geometry=args.backend == "hybrid", recognize=not args.no_recognition,
                    recovery=args.text_recovery == "auto")
            elif args.reader == "ocr":
                reader = CardTextDetector(args.ocr_model, recognize=not args.no_recognition)
            else:
                reader = NumberCardDetector(args.mode, args.min_area, not args.no_recognition)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    if args.backend in ("traditional", "text"):
        detector = reader
    elif args.backend in ("yolo", "hybrid", "armor"):
        if args.weights is None:
            parser.error(f"--backend {args.backend} 需要 --weights 指向本地权重")
        from .yolo import HybridCardDetector, YoloObjectDetector
        try:
            yolo = YoloObjectDetector(args.weights, device=args.device, image_size=args.yolo_size,
                                      confidence=args.armor_confidence if args.backend == "armor" else .25)
            if args.backend == "armor":
                from .tracking import TrackedObjectDetector
                detector = TrackedObjectDetector(yolo,label=args.armor_class)
            else:
                detector = (
                    yolo if args.backend == "yolo" else
                    HybridCardDetector(yolo, card_class=args.card_class, mode=args.mode,
                                       min_area=args.min_area, recognize=not args.no_recognition,
                                       reader=reader)
                )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    terminal_mode = "log" if args.image is not None and args.terminal == "auto" else args.terminal
    status = TerminalDashboard(_pipeline_info(args), mode=terminal_mode, interval=args.status_interval)
    if args.image is not None:
        assert detector is not None
        _offline_image(args.image, args.output_dir, detector, args.mode, status,args.preview_width)
    else:
        run_preview(args.camera, args.mode, args.min_area,
                    not args.no_recognition, detector=detector, status=status,preview_width=args.preview_width,
                    trail=args.trail == "on",
                    trail_labels=frozenset({args.armor_class}) if args.backend == "armor" else frozenset({'+'}))


if __name__ == "__main__":
    main()
