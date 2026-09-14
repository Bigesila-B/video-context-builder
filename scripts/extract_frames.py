"""
抽帧模块（帧精确 + 真实时间戳）。

两类抽帧场景：
1. 总览层（overview）：按固定间隔（默认 3 秒）均匀抽帧，用于「定位」；
2. 秒图层（second）：在某个时间窗内（默认 1 秒）密集抽帧（默认 6 帧），拼成一张「秒图」。

## 为什么不用 `-vf fps=...` ？

实测（ffmpeg 7.1 / H.264 带 B 帧）发现两个坑，都会让「抽出的帧」和「声称的时间戳」对不上：

坑 1：`-ss` 放在 `-i` 之前时，ffmpeg 丢弃「早于目标时间」的帧用的是 **DTS**，而 B 帧的 DTS 比
      PTS 滞后约 2 帧 → 实际取到的帧比目标晚约 2/帧率 秒。`-copyts` 只改标签、不改内容。
坑 2：`fps` 滤镜自身有取帧偏移 —— 即使输入起点完全正确，`fps=6` 输出的第一帧仍可能是
      第 2 帧（已用「每帧内容编码秒/帧号」的测试视频定量证实）。

## 采用的帧精确方案

    -copyts -ss <目标 - 预卷> -t <长度> -i video
    -vf "trim=start=<目标>:end=<结束>,select='not(mod(n,K))',showinfo,scale=..."
    -fps_mode passthrough

- `-copyts` + 预卷 1s：只为**提速**（从目标前 1 秒开始解码），不依赖其标签精度；
- `trim`：按**真实 PTS** 精确裁掉预卷部分，帧级精度；
- `select='not(mod(n,K))'`：`n` 从 trim 后第一帧（即恰好落在目标时间的那一帧）算起，
  因此天然自对齐；K = 帧率 × 间隔；
- `showinfo`：打印每个输出帧的 **真实 pts_time**，我们直接读它当作时间戳 ——
  不再靠「起点 + i/帧率」推算，从根上保证时间戳不说谎；
- `-fps_mode passthrough`：关掉 ffmpeg 默认的补帧（否则 image2 会把选中帧复制成 CFR，产生重复图）。

全流程 ffmpeg 流式解码，帧直接落盘，**绝不把整段视频载入内存**。
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ffmpeg_utils import ffmpeg

# (timestamp_seconds, frame_path)
Frame = Tuple[float, str]

# seek 预卷长度（秒）：只为提速，取值要 > B 帧延迟 + 一小段安全余量
SEEK_PREROLL = 1.0

# 抽帧时源帧最大宽度（缩小临时文件体积；下游 Pillow 再缩到格子尺寸）
DEFAULT_MAX_WIDTH = 960

_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9.]+)")


def _sorted_frames(paths: List[Path]) -> List[Path]:
    """按文件名排序，保证时间顺序（ffmpeg 输出是 %05d 递增，字典序即时间序）。"""
    return sorted(paths, key=lambda p: p.name)


def _safe_unlink(paths: List[Path], since: float) -> None:
    """只删除「本次运行之后才产生」的文件，避免误删用户目录里同前缀的既有文件。"""
    for p in paths:
        try:
            if p.stat().st_mtime >= since - 1e-3:
                Path(p).unlink()
        except OSError:
            pass


def _build_filter(start: float, end: Optional[float], step_frames: Optional[int], max_width: int) -> str:
    """构造帧精确滤镜链：trim -> (按帧号抽稀) -> showinfo -> 限宽。

    注意：`trim` 的多个选项之间必须用 **冒号** 分隔（`trim=start=..:end=..`），
    用逗号会被当成「滤镜链的下一个滤镜」，直接解析失败。
    """
    trim_args = f"start={max(0.0, start):.6f}"
    if end is not None:
        trim_args += f":end={end:.6f}"

    chain = [f"trim={trim_args}"]
    if step_frames and step_frames > 1:
        # n 从 trim 后第一帧算起 → 天然对齐到 start
        chain.append(f"select='not(mod(n\\,{step_frames}))'")
    chain.append("showinfo")  # 必须在 setpts/scale 之前，保证打印的是源时间轴 pts
    chain.append(f"scale='min({max_width},iw)':-2")
    return ",".join(chain)


def _run_extract(
    video_path: str,
    out_dir: str,
    prefix: str,
    vf: str,
    start: float,
    end: Optional[float],
) -> Tuple[List[str], List[float], float]:
    """执行一次抽帧，返回 (图片路径列表, 每帧真实时间戳列表, 本次开始时间戳)。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(out / f"{prefix}_%05d.jpg")
    t_start = time.time()

    def _cmd(fps_mode_args: List[str], seek: float) -> List[str]:
        args: List[str] = ["-hide_banner", "-loglevel", "info", "-y"]
        if seek > 0:
            args += ["-copyts", "-ss", f"{seek:.6f}"]
            if end is not None:
                # 限制读取长度，避免从预卷点一路读到文件尾
                args += ["-t", f"{(end - seek) + 0.5:.3f}"]
        elif end is not None:
            args += ["-t", f"{end + 0.5:.3f}"]
        args += ["-i", video_path, "-vf", vf, *fps_mode_args, "-q:v", "2", "-f", "image2", pattern]
        return args

    seek = max(0.0, float(start) - SEEK_PREROLL) if start > 0 else 0.0

    # ffmpeg >= 5 用 -fps_mode passthrough；老版本退回 -vsync 0
    proc = ffmpeg(*_cmd(["-fps_mode", "passthrough"], seek))
    if proc.returncode != 0:
        # 老版本 ffmpeg 不认 -fps_mode，清掉半成品后用 -vsync 0 重试
        _safe_unlink(list(out.glob(f"{prefix}_*.jpg")), t_start)
        proc = ffmpeg(*_cmd(["-vsync", "0"], seek))

    files = [str(p) for p in _sorted_frames(list(out.glob(f"{prefix}_*.jpg")))]
    text = (proc.stderr or "") + (proc.stdout or "")
    pts = [float(x) for x in _PTS_RE.findall(text)]
    return files, pts, t_start


def _pair(files: List[str], pts: List[float], start: float, step: float) -> List[Frame]:
    """把图片与真实时间戳配对；showinfo 解析失败时退化为「起点 + i*step」推算。"""
    frames: List[Frame] = []
    for i, fp in enumerate(files):
        if i < len(pts):
            t = pts[i]
        else:
            t = start + i * step
        frames.append((round(max(0.0, t), 3), fp))
    return frames


def extract_interval_frames(
    video_path: str,
    out_dir: str,
    interval: float = 3.0,
    src_fps: float = 25.0,
    max_width: int = DEFAULT_MAX_WIDTH,
    prefix: str = "ov",
    start: float = 0.0,
    end: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[Frame]:
    """按固定间隔均匀抽帧（总览层用）。帧精确。

    Args:
        interval: 抽帧间隔（秒），例如 3.0 表示每 3 秒 1 帧
        src_fps: 源视频帧率（用于换算成「每 K 帧取 1 帧」）
        limit: 最多抽多少帧（超出则截断，用于 token 预算兜底）
    Returns:
        [(timestamp, path), ...]，timestamp 为 showinfo 报出的真实时间（秒）
    """
    assert interval > 0, "interval 必须 > 0"
    fps = float(src_fps) if src_fps and src_fps > 0 else 25.0
    step = max(1, int(round(interval * fps)))
    vf = _build_filter(start, end, step, max_width)

    files, pts, t_start = _run_extract(video_path, out_dir, prefix, vf, start, end)
    frames = _pair(files, pts, start, interval)

    # 丢弃越过 end 的帧（trim 已大致裁好，这里做一次兜底）
    if end is not None:
        frames = [f for f in frames if f[0] <= end + 1e-6]
    if limit is not None and len(frames) > limit:
        _safe_unlink([Path(fp) for _, fp in frames[limit:]], t_start)
        frames = frames[:limit]
    return frames


def extract_window_frames(
    video_path: str,
    out_dir: str,
    start: float,
    window: float = 1.0,
    frames_in_window: int = 6,
    src_fps: float = 25.0,
    max_width: int = DEFAULT_MAX_WIDTH,
    prefix: str = "sec",
) -> List[Frame]:
    """在 [start, start+window) 内密集抽帧（秒图层用）。帧精确。

    Args:
        window: 时间窗长度，默认 1.0 秒 —— 这就是「1 秒的秒图」
        frames_in_window: 窗内要抽几帧，默认 6
    Returns:
        [(timestamp, path), ...]，时间戳来自 showinfo 的真实 pts
    """
    assert window > 0 and frames_in_window > 0
    fps = float(src_fps) if src_fps and src_fps > 0 else 25.0
    end = float(start) + float(window)

    # K = 帧率 × 窗长 / 帧数，使得窗内恰好抽到 frames_in_window 帧
    step = max(1, int(round(fps * window / frames_in_window)))
    vf = _build_filter(start, end, step, max_width)

    files, pts, t_start = _run_extract(video_path, out_dir, prefix, vf, start, end)
    frames = _pair(files, pts, start, step / fps)

    # 只保留窗内的帧，并截断到期望帧数
    frames = [f for f in frames if start - 1e-6 <= f[0] <= end + 1e-6][:frames_in_window]
    # 把本次运行产生但未被采用的图片删掉，避免污染临时目录（不会碰既有文件）
    extra = _sorted_frames(list(Path(out_dir).glob(f"{prefix}_*.jpg")))[len(frames):]
    _safe_unlink(extra, t_start)
    return frames


def extract_single_frame(
    video_path: str,
    out_path: str,
    timestamp: float,
    src_fps: float = 25.0,
    max_width: int = 1280,
) -> str:
    """抽取单帧原图（deep 模式下对「重要帧」单独发原图用）。帧精确。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vf = f"trim=start={max(0.0, timestamp):.6f},select='not(mod(n\\,1))',scale='min({max_width},iw)':-2"
    seek = max(0.0, float(timestamp) - SEEK_PREROLL) if timestamp > 0 else 0.0
    base = ["-hide_banner", "-loglevel", "error", "-y"]
    if seek > 0:
        base += ["-copyts", "-ss", f"{seek:.6f}", "-t", "1.5"]
    proc = ffmpeg(*base, "-i", video_path, "-vf", vf,
                  "-fps_mode", "passthrough", "-frames:v", "1", "-q:v", "2", out_path)
    if proc.returncode != 0:
        ffmpeg(*base, "-i", video_path, "-vf", vf,
               "-vsync", "0", "-frames:v", "1", "-q:v", "2", out_path)
    return out_path


def cleanup_dir(path: str) -> None:
    """递归删除临时目录（失败静默，不影响主流程）。"""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
