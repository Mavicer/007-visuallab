# text-pilot-v1：本轮文字区域训练候选

2026-10-15。用户已验收 P1 并授权标注、启动训练。本轮只完成一次受限试验；不是默认发布模型。

## 运行

```bash
cd ~/Desktop/GitHub/AI_Studio/Projects/007_VisualLab
uv run --extra ocr --extra yolo python -m visuallab --backend text --text-locator yolo --weights models/text-pilot-v1.pt --yolo-size 640 --full-frame-text
```

限定范围时，用 `--target-roi X Y W H` 替换 `--full-frame-text`。该入口要求权重含 `text` 类；`card-pilot-v1.pt` 和普通 COCO 权重不能用于文字定位。现有 `--backend hybrid` 命令保持原样。

`YoloTextLocator` 适配已有 `TextLocator` 接口。文本框在读取前做固定 IoU=.5 去重，再交给 `RegionTextDetector` 的灰度读取与二值诊断；不是按字符类别筛选，也不是取多个 OCR 的最高分。`--yolo-size 640` 与训练一致。文字范围不是载体真实四角，不能冒充完成透视校正或追踪。

## 数据与成绩

83 张源图，107 个审查文字框；实际数据训练 39 图/56 框、验证 16 图/26 框、测试检查 11 图/14 框。13 张训练邻近图仅回放，4 张不可靠/不完整标注图片排除。连续运动段不跨组。所有负样本都在训练组或其邻近帧；没有独立负样本留出集，不宣称复杂背景误报已解决。

单类 `text` 表示前景目标文字/符号范围，不穷尽场景所有小字。已有本地 YOLO26n 初始化，CPU，640，batch/nbs=4，冻结 10 层，最多 20 轮；13 轮早停，最佳第 8 轮，93.2 秒。没有追加训练或改动字符识别权重。

固定 conf=.25、原始文字框 IoU=.5、提案去重后：验证 P=.739/R=.654，测试检查 P=.444/R=.286。训练器 AP50≈.649、AP50–95≈.209，使用另一种阈值统计，不能与固定 P 混用。

端到端：旧 33 个可读目标 26/33；新 B 5/5、新静态纸卡 7/7，但含训练与近重复图；旧平板 A/2 为 3/6，低于 P1 DB 的 4/6；16 张运动标记为 0/16。没有达到全量替换门槛，默认保留。新增批次处理耗时中位 54.5 ms，未包含摄像头/显示，不等于实测 FPS。

权重 SHA256：`314fd3d8cff0c9ee011bdf63fe0f22931edf04aa4a11fc0967ba47045c89f356`。完整报告及逐帧证据：

`local-development/outputs/VisualLab-本轮数据审查与训练/`

下次工作先查看 `标注与训练报告.md`、`evaluation_summary.json` 和 `run_context.json`。当前无后台训练；33 项测试通过，摄像头需实际验收。不要直接追加轮数或把训练内准确率当作跨场景能力。
