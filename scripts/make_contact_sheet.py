"""
拼图（contact sheet）模块。

把一组按时间排序的帧拼成一张网格图，每小格左下角标注时间戳。
总览图与秒图共用同一套拼图逻辑，只是输入帧不同。

布局约定：layout 字符串形如 "3x2" = 3 列 x 2 行（列 x 行）。
单格尺寸：给定 --cell N 表示「每小格的**短边**不小于 N 像素」，
        长边按源视频宽高比等比放大，确保画面信息不丢。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 帧条目：(timestamp_seconds, image_path)
Entry = Tuple[float, str]

LABEL_BAR = 20          # 每格底部标签条高度（像素）
PAD = 2                 # 格与格之间的间距
BG_COLOR = (16, 16, 20)     # 拼图背景（深色，便于区分黑边）
LABEL_BG = (0, 0, 0)        # 标签条底色
LABEL_FG = (255, 214, 102)  # 标签文字颜色（暖黄，醒目）


# --------------------------------------------------------------------------- #
# 字体
# --------------------------------------------------------------------------- #
_FONT_CACHE: dict = {}


def get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """跨平台找一个可用的 TTF 字体；都找不到就退回 PIL 内置位图字体。

    时间戳标签含 ASCII 数字/冒号，内置字体也够用，所以这里不会因为字体缺失而失败。
    """
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = [
        # Windows
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font: ImageFont.ImageFont
    for c in candidates:
        if Path(c).exists():
            try:
                font = ImageFont.truetype(c, size)
                _FONT_CACHE[size] = font
                return font  # type: ignore[return-value]
            except Exception:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font  # type: ignore[assignment]
    return font  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def parse_layout(layout: str) -> Tuple[int, int]:
    """'3x2' -> (3 列, 2 行)。"""
    try:
        cols, rows = layout.lower().split("x")
        c, r = int(cols), int(rows)
        if c <= 0 or r <= 0:
            raise ValueError
        return c, r
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"非法布局 '{layout}'，应形如 '3x2'（列 x 行）") from exc


def format_timestamp(seconds: float) -> str:
    """把秒数格式化成 MM:SS；超过 1 小时用 HH:MM:SS。时间戳精确到秒。"""
    total = int(max(0.0, float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _cell_size(src_w: int, src_h: int, min_side: int) -> Tuple[int, int]:
    """按源图宽高比，算出「短边 = min_side」时的格子尺寸。"""
    if src_w <= 0 or src_h <= 0:
        return min_side, min_side
    ratio = src_w / src_h
    if ratio >= 1:      # 横图
        h = min_side
        w = int(round(min_side * ratio))
    else:               # 竖图
        w = min_side
        h = int(round(min_side / ratio))
    return max(1, w), max(1, h)


# --------------------------------------------------------------------------- #
# 核心拼图
# --------------------------------------------------------------------------- #
def build_sheet(
    entries: Sequence[Entry],
    out_path: str,
    layout: str = "4x4",
    cell: int = 224,
    title: str | None = None,
    label_every: int = 1,
) -> dict:
    """把若干帧拼成一张网格图。

    Args:
        entries: [(timestamp, image_path), ...]，需按时间升序
        out_path: 输出图片路径（.jpg / .png）
        layout: '列x行'，例如 '4x4' 表示 16 格
        cell: 每小格短边像素（不小于 224 以满足主流多模态模型的最小可读分辨率）
        title: 可选，顶部的整图标题（例如 "OVERVIEW 00:00-01:00"）
        label_every: 每隔几格标一次时间戳（1 = 每格都标）

    Returns:
        {"path": ..., "layout": ..., "cols": c, "rows": r, "cells": n,
         "sheet_width": w, "sheet_height": h, "start": t0, "end": t1}
    """
    entries = [e for e in entries if e is not None]
    if not entries:
        raise ValueError("build_sheet 收到空帧列表，无法拼图")

    cols, rows = parse_layout(layout)
    capacity = cols * rows
    entries = list(entries)[:capacity]

    # 用第一张图确定源宽高比
    with Image.open(entries[0][1]) as probe:
        src_w, src_h = probe.size
    cw, ch = _cell_size(src_w, src_h, cell)
    block_w = cw + PAD
    block_h = ch + LABEL_BAR + PAD

    # 帧数不足一整行/一整列时，把画布收缩到「实际用到的行数」，避免大片空白
    used_rows = max(1, math.ceil(len(entries) / cols))

    title_h = 34 if title else 0
    sheet_w = cols * block_w + PAD
    sheet_h = used_rows * block_h + PAD + title_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
    draw = ImageDraw.Draw(sheet)

    font = get_font(max(11, min(16, cell // 18)))
    title_font = get_font(20)

    if title:
        draw.text((10, 8), title, font=title_font, fill=(240, 240, 240))

    for idx, (ts, img_path) in enumerate(entries):
        r, c = divmod(idx, cols)
        x = PAD + c * block_w
        y = title_h + PAD + r * block_h

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                # 等比缩放到「填满格子」再居中裁剪，避免黑边、画面更满
                scale = max(cw / im.width, ch / im.height)
                nw, nh = max(1, int(im.width * scale + 0.5)), max(1, int(im.height * scale + 0.5))
                im = im.resize((nw, nh), Image.LANCZOS)
                left = (nw - cw) // 2
                top = (nh - ch) // 2
                im = im.crop((left, top, left + cw, top + ch))
                sheet.paste(im, (x, y))
        except Exception:
            # 单帧读取失败不影响整张图，用灰色占位
            draw.rectangle([x, y, x + cw, y + ch], fill=(60, 60, 60))
            draw.text((x + 6, y + 6), "N/A", font=font, fill=(200, 120, 120))

        # 标签条
        ly = y + ch
        draw.rectangle([x, ly, x + cw, ly + LABEL_BAR], fill=LABEL_BG)
        if idx % max(1, label_every) == 0:
            millis = max(0, round(float(ts) * 1000))
            frame_label = f"{format_timestamp(millis // 1000)}.{millis % 1000:03d}"
            draw.text((x + 5, ly + 3), f"#{idx + 1:03d} {frame_label}", font=font, fill=LABEL_FG)

    # 四周边框
    draw.rectangle([0, 0, sheet_w - 1, sheet_h - 1], outline=(70, 70, 80))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(out_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        sheet.save(out_path, quality=90, optimize=True)
    else:
        sheet.save(out_path)

    return {
        "path": str(out_path),
        "layout": f"{cols}x{rows}",
        "cols": cols,
        "rows": rows,
        "cells": len(entries),
        "sheet_width": sheet_w,
        "sheet_height": sheet_h,
        "start": round(entries[0][0], 3),
        "end": round(entries[-1][0], 3),
    }


def main() -> int:
    """自检入口：把某个目录下的图片拼成一张总览图。"""
    import argparse

    ap = argparse.ArgumentParser(description="把一批图片拼成 contact sheet")
    ap.add_argument("frame_dir", help="包含 jpg/png 的目录")
    ap.add_argument("out", help="输出图片路径")
    ap.add_argument("--layout", default="4x4")
    ap.add_argument("--cell", type=int, default=224)
    ap.add_argument("--interval", type=float, default=1.0, help="相邻帧的时间间隔（用于生成时间戳标签）")
    args = ap.parse_args()

    files = sorted(
        [p for p in Path(args.frame_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: p.name,
    )
    entries: List[Entry] = [(i * args.interval, str(p)) for i, p in enumerate(files)]
    info = build_sheet(entries, args.out, layout=args.layout, cell=args.cell, title=args.frame_dir)
    print(f"已生成 {info['path']}  {info['sheet_width']}x{info['sheet_height']}  {info['cells']} 格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
