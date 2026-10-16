"""Prepare the single P1 text detector; runtime never downloads models."""
from pathlib import Path
import hashlib
import tempfile
import urllib.request

URL = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx'
SHA256 = 'd2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9'
TARGET = Path(__file__).resolve().parents[1] / 'models/ch_PP-OCRv4_det_mobile.onnx'


def main() -> None:
    if TARGET.exists():
        if hashlib.sha256(TARGET.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError(f'已有模型校验失败：{TARGET}')
        print(f'文字定位模型已就绪：{TARGET}')
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=TARGET.parent, suffix='.download', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('模型 SHA256 校验失败')
        temporary.replace(TARGET)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'文字定位模型已保存并校验：{TARGET}')


if __name__ == '__main__':
    main()
