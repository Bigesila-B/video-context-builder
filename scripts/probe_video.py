"""
视频元数据探测（probe）。

优先用 ffprobe 读取结构化 JSON；若系统无 ffprobe（例如只装了 imageio-ffmpeg），
则退化为解析 `ffmpeg -i` 的 stderr 文本。两条路径都返回同一套字段。

用法：
    python scripts/probe_video.py input.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 允许以脚本方式直接运行（把 scripts/ 加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffmpeg_utils import ffmpeg, ffprobe_bin, run  # noqa: E402


def _fraction_to_float(value: Any) -> float:
    """把 '30000/1001' 这类分数字符串转成浮点数。"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    if "/" in s:
        num, den = s.split("/", 1)
        try:
            return float(num) / float(den) if float(den) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _probe_with_ffprobe(path: str) -> Dict[str, Any]:
    """用 ffprobe 读取元数据（首选路径，字段最准）。"""
    proc = run(
        [
            ffprobe_bin(),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path,
        ]
    )
    data = json.loads(proc.stdout or "{}")
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for candidate in (
        (video or {}).get("duration"),
        fmt.get("duration"),
        (fmt.get("tags") or {}).get("DURATION"),
    ):
        d = _fraction_to_float(candidate)
        if d > 0:
            duration = d
            break

    fps = 0.0
    if video:
        fps = _fraction_to_float(video.get("avg_frame_rate")) or _fraction_to_float(video.get("r_frame_rate"))

    nb_frames = 0
    if video and video.get("nb_frames"):
        try:
            nb_frames = int(video["nb_frames"])
        except (TypeError, ValueError):
            nb_frames = 0
    if nb_frames == 0 and fps > 0 and duration > 0:
        nb_frames = int(round(fps * duration))

    return {
        "path": str(path),
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "width": int((video or {}).get("width") or 0),
        "height": int((video or {}).get("height") or 0),
        "has_audio": audio is not None,
        "video_codec": (video or {}).get("codec_name", "") or "",
        "audio_codec": (audio or {}).get("codec_name", "") or "",
        "nb_frames": nb_frames,
        "bit_rate": int(_fraction_to_float(fmt.get("bit_rate"))),
        "probe_source": "ffprobe",
    }


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_SIZE_RE = re.compile(r",\s*(\d{2,5})x(\d{2,5})[\s,]")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?: Video:")
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?: Audio:")


def _probe_with_ffmpeg(path: str) -> Dict[str, Any]:
    """退路：解析 `ffmpeg -i` 的 stderr（无 ffprobe 时使用）。"""
    proc = ffmpeg("-hide_banner", "-i", path)
    text = (proc.stderr or "") + (proc.stdout or "")

    duration = 0.0
    m = _DUR_RE.search(text)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = h * 3600 + mi * 60 + s

    # 视频流所在行（用于取分辨率与帧率）
    video_line = ""
    for line in text.splitlines():
        if _VIDEO_RE.search(line):
            video_line = line
            break

    width = height = 0
    if video_line:
        ms = _SIZE_RE.search(video_line)
        if ms:
            width, height = int(ms.group(1)), int(ms.group(2))

    fps = 0.0
    if video_line:
        mf = _FPS_RE.search(video_line)
        if mf:
            fps = float(mf.group(1))

    has_audio = bool(_AUDIO_RE.search(text))

    return {
        "path": str(path),
        "duration": round(duration, 3),
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "has_audio": has_audio,
        "video_codec": "",
        "audio_codec": "",
        "nb_frames": int(round(fps * duration)) if fps and duration else 0,
        "bit_rate": 0,
        "probe_source": "ffmpeg-fallback",
    }


def probe_video(video_path: str) -> Dict[str, Any]:
    """探测视频元数据。

    Returns:
        dict，至少包含 duration / fps / has_audio / width / height。

    Raises:
        FileNotFoundError: 输入文件不存在。
    """
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    if ffprobe_bin():
        try:
            meta = _probe_with_ffprobe(str(p))
            if meta["duration"] > 0:
                return meta
        except Exception:
            pass  # ffprobe 失败则走 ffmpeg 兜底

    meta = _probe_with_ffmpeg(str(p))
    if meta["duration"] <= 0:
        raise RuntimeError(
            f"无法解析视频时长：{video_path}（文件可能损坏或不是受支持的媒体格式）"
        )
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="探测视频元数据")
    ap.add_argument("video", help="视频文件路径")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args()

    meta = probe_video(args.video)
    if args.json:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    else:
        print(f"路径      : {meta['path']}")
        print(f"时长      : {meta['duration']:.2f} s")
        print(f"帧率      : {meta['fps']} fps")
        print(f"分辨率    : {meta['width']}x{meta['height']}")
        print(f"音轨      : {'有' if meta['has_audio'] else '无'}")
        print(f"总帧数    : {meta['nb_frames']}")
        print(f"探测方式  : {meta['probe_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
