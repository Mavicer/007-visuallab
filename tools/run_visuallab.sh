#!/bin/zsh
# 从任意目录启动；切换目录只影响本脚本，不改变用户终端的位置。
set -eu
project_dir=${0:A:h:h}
cd -- "$project_dir"

uv_bin=$(command -v uv || true)
if [[ -z "$uv_bin" && -x /opt/homebrew/bin/uv ]]; then
    uv_bin=/opt/homebrew/bin/uv
fi
if [[ -z "$uv_bin" ]]; then
    print -u2 -- '未找到 uv，请先安装 uv 并重新打开终端。'
    exit 127
fi

# `run visuallab armor` selects the separate whole-marker model. Ordinary
# `run visuallab` keeps general ASCII OCR; neither profile loads both models.
if [[ "${1:-}" == armor ]]; then
    shift
    armor_weights=models/armor-v1.pt
    [[ -f "$armor_weights" ]] || armor_weights=weights/armor-v1.pt
    exec "$uv_bin" run --extra yolo python -m visuallab \
        --backend armor --weights "$armor_weights" --yolo-size 640 "$@"
fi

# 默认沿用当前 YOLO 文字定位候选；显式指定 ROI 时不再全画面扫描。
scope_args=(--full-frame-text)
for arg in "$@"; do
    if [[ "$arg" == --target-roi || "$arg" == --target-roi=* ]]; then
        scope_args=()
        break
    fi
done

# 原样转交额外参数，路径中的空格不会被拆开。
text_weights=models/text-pilot-v1.pt
[[ -f "$text_weights" ]] || text_weights=weights/text-pilot-v1.pt
exec "$uv_bin" run --extra ocr --extra yolo python -m visuallab \
    --backend text --text-locator yolo \
    --weights "$text_weights" --yolo-size 640 \
    --text-recovery auto "${scope_args[@]}" "$@"
