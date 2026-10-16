# 本地通用 ASCII OCR

当前命令行默认 `--reader ocr`。`--reader template` 仅是原先 1/2 模板基线，不能用于新增字符验收。

采用一份预训练英文 PP-OCRv4 CTC 识别模型，直接使用 ONNX Runtime CPU 推理，避免额外安装会覆盖桌面 OpenCV 的 headless 依赖。

- 发布方与地址：`https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx`
- 官方清单：<https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml>
- SHA256：`e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318`
- 本地路径：`models/en_PP-OCRv4_rec_mobile.onnx`，不入 Git。
- 安装：`uv sync --extra ocr`；首次模型准备：`uv run --extra ocr python tools/setup_ocr.py`。

`text_ocr.py` 检查模型内嵌字符表覆盖 ASCII 32～126，并验证其与 97 维 CTC 输出匹配。空格、CTC 空白符和真实问号各有不同含义；显示用 `is_unknown` 标记失败，避免把真正的 `?` 当未知结果。没有向模型加数字白名单或逐字符模板。

纸面定位与 OCR 解耦。纸片四角可信时做透视校正，纸内墨迹整体提取为高对比视图、裁去多余留白并加白边，再保持比例缩放到模型输入。句点、短横、冒号及字母的分离部分不再受旧数字形状门槛过滤。当前一个纸面作为单字符/单行处理；多行排版、90/180 度方向识别与镜像还需独立模块。

字符表支持不代表所有外观均可准确识别。两种印刷字体的 20 个单数字全部读对，大小写字母行也读对；188 个孤立非空格 ASCII 中 157 个准确，混淆集中在形状相似字符和标点。实拍 `3`、`4`、`7` 也有读对的例子，但整帧结果仍受纸片定位、遮挡、模糊影响。

ASCII 控制码没有可识别字形；空白纸张不作为一个空格输出。OCR 分数为模型输出分数，不是正确概率。模型文件与基准失败清单见本轮采集审查报告。
