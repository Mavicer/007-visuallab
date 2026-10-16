# P1：对象范围、文字范围与 OCR 输入

2026-10-15。当前为**可选验收功能**，没有改变原有默认后端或读取器。

## 数据流与接口

`对象 ROI -> TextLocator.locate -> TextRegion.quad -> 灰度读取 + 二值诊断 -> Observation`

- `--backend text` 的对象范围来自 `--target-roi X Y W H`（画面比例）；范围外不做文字搜索。
- `--text-layout direct` 跳过文字定位器，直接读取选定的单字符/单行范围。孤立标点、大字符可能被文本行检测器漏掉，所以此入口独立保留。
- `--full-frame-text` 必须显式指定，仅适合离线版面诊断；屏幕 UI、书本小字也是文字，检测器无法判断它们是否是用户想要的目标。
- `--backend hybrid --reader located-ocr`：对象框仍由已有 YOLO `card` 类提供；可信纸角先做现有透视变换，再定位文字。没有卡片框时不扫描背景。它仍会受纸片模型漏检限制。
- 新模块 `visuallab/text_regions.py` 提供可注入的 `TextLocator`、`TextReader`。替换上游对象定位器或读取器不需要把字符类别写进 YOLO。
- 自动文字框只提供文字区域近似几何，不保证找到了真实载体四角。直接 ROI 的框和中心是**用户选定范围**，不是字符精确位置，也不是追踪 ID；未来云台控制不能把它当作已完成目标定位。

## 运行

```bash
cd ~/Desktop/GitHub/AI_Studio/Projects/007_VisualLab
uv sync --extra ocr --extra yolo
# 首次准备候选定位器，已下载的模型会校验后复用
uv run --extra ocr --extra yolo python tools/setup_text_detector.py

# 先验收 OCR：把完整单字符/单行放进画面中央区域，留少量空白
uv run --extra ocr --extra yolo python -m visuallab --backend text --text-layout direct --target-roi 0.25 0.15 0.5 0.7

# 同一范围内尝试自动文本定位；不要求对象是白纸
uv run --extra ocr --extra yolo python -m visuallab --backend text --target-roi 0.25 0.15 0.5 0.7

# 卡片模式中的可选读取器；该候选仍存在拒识/重复框，尚不推荐替换原命令
uv run --extra ocr --extra yolo python -m visuallab --backend hybrid --weights models/card-pilot-v1.pt --yolo-size 416 --reader located-ocr

# 原有启动命令（保持不变）
uv run --extra ocr --extra yolo python -m visuallab --backend hybrid --weights models/card-pilot-v1.pt --yolo-size 416

# 单帧调试；输出 JSON、结果图、灰度/二值图、实际 OCR 裁剪
uv run --extra ocr --extra yolo python -m visuallab --backend text --full-frame-text --image captures/某张原始帧.png --output-dir runs/text-preview
```

`Gray/Binary/Recognition` 三窗口继续使用。实时仍按 S 保存原始帧；详细裁剪和诊断通过离线命令导出。`--no-recognition` 只定位，不加载字符读取器。

## 输入和接受规则

每个文本范围最多使用两个固定输入：原始灰度、Otsu 二值图。灰度为实际输出，分数至少 .6 且内容非空才接受；二值结果只用于诊断，不凭最高分覆盖灰度，也不删除额外的 `|/-` 来提高成绩。分歧在 `result.json` 的 `diagnostics.inputs` 中记录。`.6` 是沿用的工程阈值，分数不是正确概率。

DB 图固定阈值 .3、区域均分 .7、unclip 比例 2、最小边 3，输入最长边 960、32 对齐；文字区域另留通用 12% 单侧余量，没有字符专用分支。识别器不变，无新增训练。灰度/二值对照每区域最多两次调用，额外耗时已单独测量。

## 候选模型、来源与许可

- 定位模型：`ch_PP-OCRv4_det_mobile.onnx`，约 4.5 MiB。名称中的 `ch` 指检测模型来源；它只定位文字，英文识别器的字符范围仍未扩展到中文。
- 固定来源：[RapidAI/RapidOCR v3.10.0 模型](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx)。
- SHA256：`d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9`。
- 预处理与 DB 后处理参考 [RapidOCR v3.10.0 utils.py](https://github.com/RapidAI/RapidOCR/blob/v3.10.0/python/rapidocr/ch_ppocr_det/utils.py)，Copyright (c) 2021 RapidOCR Authors，Apache-2.0；许可全文保存在 `docs/licenses/RapidOCR-APACHE-2.0.txt`。本文实现为精简适配版，矩形扩张使用等距偏移后的外接矩形关系，省去 Shapely/Pyclipper；使用最长边上限及额外裁剪余量。
- 复用既有 ONNXRuntime，不安装 RapidOCR 整包、额外深度学习框架或第二套 OpenCV。模型只在准备脚本下载，运行时本地推理。

## 本轮验收结论

全部 52 帧已回放。6 个平板目标：原混合路径 0/6，全画面 DB 4/6，人工指定 ROI 直接读取 6/6；后者**不代表自动定位已经成功**。6 个空白纸巾/手机负样本在两个新自动路径中均无接受文字。旧 33 个可读目标：原混合 13/33，可选卡片文字定位 18/33，全画面 DB 24/33；这是旧场景诊断集，含训练帧，不是独立测试成绩。

新 5 个正向纸卡数字：原路径 4/5；全画面 DB 4/5，但成功帧互有变化；卡片文字定位在第 4 帧拒识，第 5 帧重复输出，尚不能替换原路径。188 个 ASCII 原始预测逐项一致，仍为 157/188；其中实际接受且正确 149/188。字符表覆盖不等于所有字形都能读对。

同机 CPU 顺序回放，不含模型加载、摄像头/显示：旧混合中位 50.4 ms，全画面 DB 96.2 ms，可选卡片文字定位 102.8 ms；直接 ROI 六帧中位 27.6 ms。自动候选尚未通过完整门槛，保留原默认，暂停于 P1 验收；P2 的跨帧 ID、择帧与历史状态尚未实施。
