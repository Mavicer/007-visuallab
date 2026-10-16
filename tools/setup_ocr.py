"""Download one pinned public OCR model; recognition thereafter is offline."""
from pathlib import Path
import hashlib
import tempfile
import urllib.request

URL = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx'
SHA256 = 'e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318'
TARGET = Path(__file__).resolve().parents[1] / 'models/en_PP-OCRv4_rec_mobile.onnx'


def main() -> None:
    if TARGET.exists():
        if hashlib.sha256(TARGET.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError(f'已有模型校验失败，请检查文件：{TARGET}')
        print(f'OCR 模型已就绪：{TARGET}')
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=TARGET.parent, suffix='.download', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('模型 SHA256 校验失败，下载结果不会投入使用')
        temporary.replace(TARGET)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'OCR 模型已保存并校验：{TARGET}')


if __name__ == '__main__':
    main()
