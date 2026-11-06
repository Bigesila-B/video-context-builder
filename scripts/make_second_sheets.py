"""
秒图（second sheet）模块 —— 本 skill 的核心能力。

「秒图」= 在 1 秒内抽取多帧（默认 6 帧），拼成一张 3x2 的图。
这样模型看一张图，实际上看到了这 1 秒内的连续动作分解。

关键点：**不对整个长视频全量生成秒图**（那会 token 爆炸）。
只对「关键时间段」生成：场景切换、动作剧烈、语音活动、用户指定区间。

本模块还负责关键段检测（detect_key_seconds）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# 允许以 `python scripts/xxx.py` 直接运行：先把 scripts/ 加入搜索路径，再导入同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_frames import extract_window_frames  # noqa: E402
from ffmpeg_utils import ffmpeg  # noqa: E402
from make_contact_sheet import Entry, build_sheet, format_timestamp  # noqa: E402

# --------------------------------------------------------------------------- #
# 关键段检测
# --------------------------------------------------------------------------- #
_SCENE_RE = re.compile(r"pts_time:([0-9.]+)")
_SCORE_RE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def detect_scene_changes(
    video_path: str,
    threshold: float = 0.30,
    start: float = 0.0,
    duration: float | None = None,
) -> List[float]:
    """用 ffmpeg 的 scene 滤镜检测镜头切换点，返回时间戳（秒）列表。

    原理：`select='gt(scene,THRESHOLD)'` 让 ffmpeg 只输出「与前一帧差异超过阈值」的帧，
    同时 `metadata=print` 把这些帧的 pts_time 和 scene_score 打到 stderr。
    scene_score 越大说明画面变化越剧烈 —— 可用来近似「动作剧烈」程度。

    该方法零额外依赖（不需要 OpenCV），且是流式解码、不吃内存。
    """
    cmd: List[str] = ["-hide_banner", "-loglevel", "info"]
    if start and start > 0:
        # 与 extract_frames 同样的「预卷 + trim」策略：避免 B 帧 DTS 延迟导致时间戳偏移
        seek = max(0.0, float(start) - 1.0)
        cmd += ["-copyts", "-ss", f"{seek:.3f}"]
        if duration is not None and duration > 0:
            cmd += ["-t", f"{(duration + 1.5):.3f}"]
    elif duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    trim = f"trim=start={max(0.0, float(start)):.3f}"
    if duration is not None and duration > 0:
        trim += f":end={float(start) + duration:.3f}"
    cmd += [
        "-i", video_path,
        "-an",
        "-vf", f"{trim},select='gt(scene,{threshold})',metadata=print",
        "-f", "null", "-",
    ]
    try:
        proc = ffmpeg(*cmd)
    except Exception:
        return []

    text = (proc.stderr or "") + "\n" + (proc.stdout or "")

    times: List[float] = []
    pending_pts: float | None = None
    for line in text.splitlines():
        m = _SCENE_RE.search(line)
        if m:
            pending_pts = float(m.group(1))
            continue
        ms = _SCORE_RE.search(line)
        if ms and pending_pts is not None:
            times.append(round(pending_pts, 3))
            pending_pts = None

    if not times:
        # 某些 ffmpeg 版本不打印 scene_score，退回「只抓 pts_time」
        times = [round(float(x), 3) for x in _SCENE_RE.findall(text)]

    # 去重 + 升序
    return sorted({t for t in times if t >= 0})


def speech_active_seconds(segments: Sequence[dict]) -> List[float]:
    """从 STT 的 segments 里提取「有语音活动」的秒（每段的起始秒）。"""
    out: List[float] = []
    for seg in segments or []:
        if seg.get("timestamp_source") == "chunk":
            continue
        try:
            out.append(round(float(seg.get("start", 0.0)), 3))
        except (TypeError, ValueError):
            continue
    return out


def pick_key_seconds(
    duration: float,
    budget_frames: int,
    frames_per_second_sheet: int = 6,
    scene_changes: Iterable[float] = (),
    speech_starts: Iterable[float] = (),
    user_range: Tuple[float, float] | None = None,
    min_gap: float = 1.0,
) -> List[float]:
    """挑选要生成秒图的关键秒。

    策略（按优先级）：
      1. 场景切换秒（画面变化大 -> 信息量大）；
      2. 语音活动秒（有讲解/对白 -> 内容密度高）；
      3. 均匀补点（若预算还有富余，按时间等距铺开保证覆盖面）。

    最后按「预算帧数 / 每秒帧数」决定最多能出多少张秒图，并做最小间隔去重。

    Args:
        budget_frames: 秒图部分可用的总帧数预算
        user_range: (start, end) 用户指定区间；给了就只在该区间内挑点
    Returns:
        升序的关键秒列表（每个元素是一张秒图的起点）
    """
    max_sheets = max(0, budget_frames // max(1, frames_per_second_sheet))
    if max_sheets <= 0:
        return []

    lo, hi = (0.0, duration)
    if user_range is not None:
        lo, hi = max(0.0, user_range[0]), min(duration, user_range[1])
    if hi <= lo:
        return []

    ordered: List[float] = []
    seen: List[float] = []

    def _accept(t: float) -> bool:
        """与已选点保持 min_gap 的最小间隔，避免连续重复抽同一段。"""
        if t < lo - 1e-6 or t > hi - 1e-6:
            return False
        for s in seen:
            if abs(s - t) < min_gap:
                return False
        seen.append(t)
        ordered.append(round(t, 3))
        return True

    # 1) 场景切换优先。数量超过名额时按步长抽稀，保证铺满整条时间线而不是挤在开头
    sc = sorted(scene_changes)
    if len(sc) > max_sheets:
        stride = len(sc) / max_sheets
        sc = [sc[min(len(sc) - 1, int(i * stride))] for i in range(max_sheets)]
    for t in sc:
        if len(seen) >= max_sheets:
            break
        _accept(t)

    # 2) 语音活动次之（同样抽稀）
    sp = sorted(speech_starts)
    if len(sp) > max_sheets:
        stride = len(sp) / max_sheets
        sp = [sp[min(len(sp) - 1, int(i * stride))] for i in range(max_sheets)]
    for t in sp:
        if len(seen) >= max_sheets:
            break
        _accept(t)

    # 3) 均匀补点
    if len(seen) < max_sheets:
        span = hi - lo
        step = span / max_sheets
        for i in range(max_sheets):
            if len(seen) >= max_sheets:
                break
            _accept(lo + i * step + step / 2)

    return sorted(ordered)[:max_sheets]


# --------------------------------------------------------------------------- #
# 生成秒图
# --------------------------------------------------------------------------- #
def build_second_sheets(
    video_path: str,
    out_dir: str,
    seconds: Sequence[float],
    frames_per_sheet: int = 6,
    window: float = 1.0,
    layout: str = "3x2",
    cell: int = 224,
    src_fps: float = 25.0,
    max_width: int = 960,
    label: bool = True,
    end_time: float | None = None,
) -> List[Dict]:
    """为每个关键秒生成一张秒图。

    Args:
        seconds: 关键秒起点列表（来自 pick_key_seconds）
        frames_per_sheet: 每张秒图抽多少帧（默认 6，与 3x2 布局正好匹配）
        window: 秒图覆盖的时间窗（默认 1.0 秒 —— 这就是「秒图」的定义）
        layout: 默认 3x2（列 x 行 = 6 格）
        cell: 每小格短边像素，默认 224
        src_fps: 源视频帧率，用于把「窗内抽 N 帧」换算成「每 K 帧取 1 帧」
    Returns:
        [{path, start, end, frames, layout, frame_timestamps, ...}, ...]，
        即输出 JSON 的 second_sheets 字段
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    for i, sec in enumerate(seconds):
        actual_window = min(window, end_time - float(sec)) if end_time is not None else window
        if actual_window <= 0:
            continue
        frames = extract_window_frames(
            video_path,
            str(out),
            start=float(sec),
            window=actual_window,
            frames_in_window=int(frames_per_sheet),
            src_fps=src_fps,
            max_width=max_width,
            prefix=f"s{i:04d}",
        )
        if not frames:
            continue

        entries: List[Entry] = [(t, p) for t, p in frames]
        sheet_path = out / f"second_{i:04d}_{int(sec)}s.jpg"
        title = f"DETAIL SHEET  {format_timestamp(sec)} - {format_timestamp(sec + actual_window)}  " \
                f"({len(entries)} frames / {actual_window:g}s)"
        info = build_sheet(
            entries,
            str(sheet_path),
            layout=layout,
            cell=cell,
            title=title if label else None,
        )

        results.append({
            "path": str(sheet_path),
            "start": round(float(sec), 3),
            "end": round(float(sec + actual_window), 3),
            "frames": len(entries),
            "layout": info["layout"],
            "timestamp": format_timestamp(sec),
            "frame_timestamps": [round(t, 3) for t, _ in frames],
            "sheet_width": info["sheet_width"],
            "sheet_height": info["sheet_height"],
        })

    return results


def build_second_sheets_for_range(
    video_path: str,
    out_dir: str,
    start: float,
    end: float,
    step: float = 1.0,
    **kwargs,
) -> List[Dict]:
    """custom-range 模式：对指定区间逐秒生成秒图。"""
    seconds = []
    t = float(start)
    while t < float(end) - 1e-6:
        seconds.append(round(t, 3))
        t += max(0.1, step)
    return build_second_sheets(video_path, out_dir, seconds, **kwargs)


def main() -> int:
    """自检入口：对某个时间点生成一张秒图。"""
    import argparse

    ap = argparse.ArgumentParser(description="生成关键秒图（1 秒内多帧拼成一张）")
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--seconds", default="5,12,30", help="逗号分隔的关键秒起点")
    ap.add_argument("--frames", type=int, default=6, help="每张秒图的帧数")
    ap.add_argument("--layout", default="3x2")
    ap.add_argument("--cell", type=int, default=224)
    args = ap.parse_args()

    secs = [float(x) for x in args.seconds.split(",") if x.strip()]
    sheets = build_second_sheets(
        args.video, args.out_dir, secs,
        frames_per_sheet=args.frames, layout=args.layout, cell=args.cell,
    )
    for s in sheets:
        print(f"{s['timestamp']}  {s['path']}  {s['layout']}  {s['frames']}帧  {s['sheet_width']}x{s['sheet_height']}")
    print(f"共 {len(sheets)} 张秒图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
