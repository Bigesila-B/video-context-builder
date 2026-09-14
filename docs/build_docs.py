"""
文档配图的生成脚本。

做三件事：
1. 读取 docs/*.svg（浅色版配色，透明背景）；
2. 用同一套几何生成 **深色版** SVG（只换配色，不动布局）；
3. 用无头浏览器把两版都渲染成 2 倍分辨率 PNG，供 README 用
   `<picture>` + `prefers-color-scheme` 自动切换深浅主题。

用法：
    python docs/build_docs.py

依赖：本机装了 Edge 或 Chrome 之一（脚本会自动找）。
找不到浏览器时只生成 SVG，不报错。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 浅色 -> 深色 配色映射（取自已验证的 9 色阶：50/100/200/400/600/800/900）
# 说明：单次 re.sub 不会二次扫描替换结果，所以互为目标的映射（如灰 200 <-> 灰 600）安全。
DARK_MAP = {
    "#F1EFE8": "#2C2C2A",   # 灰 50  填充 -> 灰 900
    "#5F5E5A": "#B4B2A9",   # 灰 600 描边 -> 灰 200
    "#2C2C2A": "#D3D1C7",   # 灰 900 标题 -> 灰 100
    "#444441": "#B4B2A9",   # 灰 800 副标题 -> 灰 200
    "#E6F1FB": "#0C447C",   # 蓝 50  填充 -> 蓝 800
    "#185FA5": "#85B7EB",   # 蓝 600 描边 -> 蓝 200
    "#0C447C": "#B5D4F4",   # 蓝 800 标题 -> 蓝 100
    "#FAEEDA": "#633806",   # 琥珀 50  填充 -> 琥珀 800
    "#854F0B": "#EF9F27",   # 琥珀 600 描边 -> 琥珀 200
    "#633806": "#FAC775",   # 琥珀 800 标题 -> 琥珀 100
    "#A32D2D": "#F09595",   # 红 600 -> 红 200
    "#0F6E56": "#5DCAA5",   # 青 600 -> 青 200
    "#888780": "#B4B2A9",   # 灰 400 箭头 -> 灰 200
    "#B4B2A9": "#5F5E5A",   # 网格线：灰 200 -> 灰 600
    "#FFFFFF": "#0D1117",   # 文字遮罩底色 -> 深色页面底（与背景色一致）
}

BG_LIGHT = "#FFFFFF"
BG_DARK = "#0D1117"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


def find_browser() -> str | None:
    for p in BROWSERS:
        if os.path.exists(p):
            return p
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def to_dark(svg: str) -> str:
    """只替换颜色，不动任何几何与文案。"""
    return re.sub(
        r"#[0-9A-Fa-f]{6}",
        lambda m: DARK_MAP.get(m.group(0).upper(), m.group(0)),
        svg,
    )


def with_background(svg: str, color: str) -> str:
    """在 SVG 首个子元素位置插入一个铺满 viewBox 的纯色背景矩形。"""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if not m:
        return svg
    w, h = m.group(1), m.group(2)
    rect = f'<rect x="0" y="0" width="{w}" height="{h}" fill="{color}"/>'
    # 插到 <title> 之后（title 必须是首个子元素，否则可访问性失效）
    tm = re.search(r"</title>", svg)
    if tm:
        return svg[: tm.end()] + rect + svg[tm.end():]
    return re.sub(r"(<svg[^>]*>)", r"\1" + rect, svg, count=1)


def render_png(browser: str, svg_markup: str, out_png: Path, scale: int = 2) -> bool:
    """用无头浏览器把 SVG 渲染成 PNG。"""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_markup)
    if not m:
        return False
    w, h = int(float(m.group(1)) * scale), int(float(m.group(2)) * scale)

    body = svg_markup.replace(
        "<svg ",
        f'<svg style="display:block;width:{w}px;height:{h}px" ',
        1,
    )
    html = (
        "<!doctype html><meta charset=\"utf-8\">"
        "<style>html,body{margin:0;padding:0;overflow:hidden}</style>" + body
    )

    with tempfile.TemporaryDirectory(prefix="vc_docs_") as td:
        html_path = Path(td) / "page.html"
        html_path.write_text(html, encoding="utf-8")
        profile = Path(td) / "prof"
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--user-data-dir={profile}",
            "--virtual-time-budget=5000",
            f"--window-size={w},{h}",
            f"--screenshot={out_png}",
            html_path.as_uri(),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=90)
        except Exception as exc:  # noqa: BLE001
            print(f"    渲染异常：{exc}")
            return False
    return out_png.exists() and out_png.stat().st_size > 0


def main() -> int:
    browser = find_browser()
    print(f"浏览器：{browser or '（未找到，只生成 SVG）'}")

    svgs = sorted(HERE.glob("*.svg"))
    svgs = [p for p in svgs if not p.stem.endswith("-dark")]
    if not svgs:
        print("docs/ 下没有找到 SVG")
        return 1

    for svg_path in svgs:
        light_svg = svg_path.read_text(encoding="utf-8")
        dark_svg = to_dark(light_svg)
        (HERE / f"{svg_path.stem}-dark.svg").write_text(dark_svg, encoding="utf-8")

        print(f"  {svg_path.name}")
        print(f"    -> {svg_path.stem}-dark.svg")

        if not browser:
            continue
        pairs = [
            (with_background(light_svg, BG_LIGHT), HERE / f"{svg_path.stem}.png"),
            (with_background(dark_svg, BG_DARK), HERE / f"{svg_path.stem}-dark.png"),
        ]
        for markup, out in pairs:
            ok = render_png(browser, markup, out)
            size = f"{out.stat().st_size // 1024} KB" if ok else "失败"
            print(f"    -> {out.name}  {size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
