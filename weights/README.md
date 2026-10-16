# 实验权重

公开副本只清理本机训练路径元数据，参数张量逐项一致；本地 models/ 原件保留。均为小样本试验，不是通用装甲板或任意 ASCII 性能保证。派生 YOLO 权重使用 AGPL-3.0。

| 文件 | 用途 | 默认阈值 |
|---|---|---|
| text-pilot-v1.pt | 单类 text，需再接 OCR；不按具体字符训练 YOLO 类别 | .25 |
| armor-v1.pt | 单类 armor，现有加号与四角图案的整块打印模拟标记 | .5 |

armor 从 text-pilot-v1 微调，训练55图/24正目标、31负；验证41图/13正、28负；检查27图/10正、17负。最多20轮，17轮早停，最佳12轮。640、batch/nbs4、冻结10层、AdamW、CPU；旋转/亮度/小幅透视增强，无镜像/mosaic。原始含人像采集不公开，不能仅凭本仓库精确复现训练数据。

文字模型的上游初始化为 YOLO26n。OCR 模型不内置，请运行 tools/setup_ocr.py；程序推理后续在本地进行。仅加载可信来源的 PyTorch 权重。

## 公开文件 SHA256

- `text-pilot-v1.pt`: `0e5c0605b2e8faf319eb715ecf1bacab401232af2c9654efba062d1099a13502`
- `armor-v1.pt`: `4f23cded15a99ae54ab49cbd234db4bcd1e4d72ee85fca6ded50fc910bf423cb`
