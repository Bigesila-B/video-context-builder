"""
video-context-builder 主编排器。

把视频转换成「多模态模型可消费的视频上下文包」：
  视频元数据 → 总览拼图 → 关键秒图 → 可选 STT → 目标模型提示词 → token 估算与超限提醒

CLI:
    python scripts/build_video_context.py --video input.mp4 --mode auto --stt \
        --max-tokens 50000 --out out.json

Python:
    from scripts.build_video_context import build_video_context
    ctx = build_video_context("input.mp4", mode="auto", stt=True, max_tokens=50000)

核心设计：
- **分层**：总览层负责「定位」，关键秒层负责「看清」。默认不对整段长视频全量生成秒图。
- **预算**：处理前先估算图像/文本 token；超限时自动降级并**明确提醒**，绝不静默塞几百张图。
- **边界**：本工具只生产「材料 + 提示词 + 预算」，不调用目标模型；长视频的 map-reduce 只给出分段计划与提示词。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 允许脚本直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_frames import (  # noqa: E402
    cleanup_dir,
    extract_interval_frames,
    extract_single_frame,
    extract_window_frames,
)
from ffmpeg_utils import FFmpegNotFoundError  # noqa: E402
from make_contact_sheet import (  # noqa: E402
    LABEL_BAR,
    PAD,
    _cell_size,
    build_sheet,
    format_timestamp,
    parse_layout,
)
from make_second_sheets import (  # noqa: E402
    build_second_sheets,
    detect_scene_changes,
    pick_key_seconds,
    speech_active_seconds,
)
from probe_video import probe_video  # noqa: E402
from prompt_builder import (  # noqa: E402
    build_map_prompt,
    build_prompt,
    build_reduce_prompt,
)
from token_estimator import budget_check, estimate_image_tokens, estimate_pack, estimate_text_tokens  # noqa: E402
from transcribe import transcribe  # noqa: E402

# --------------------------------------------------------------------------- #
# 模式档案
# --------------------------------------------------------------------------- #
# overview_cap  : 总览层允许的最大采样帧数
# second_cap    : 秒图层允许的最大帧数（0 = 不生成秒图）
# stt_default   : 该模式默认是否开 STT
MODE_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "overview_interval": 4.0,
        "overview_layout": "4x4",
        "overview_cap": 32,
        "second_cap": 0,
        "stt_default": False,
        "max_frames": 32,
        "originals": 0,
        "desc": "总览图 1-2 张，最多 32 帧，默认无 STT（只想知道视频讲了什么）",
    },
    "standard": {
        "overview_interval": 3.0,
        "overview_layout": "4x4",
        "overview_cap": 32,
        "second_cap": 64,
        "stt_default": True,
        "max_frames": 96,
        "originals": 0,
        "desc": "总览图 + 关键秒图，最多 96 帧，默认开 STT（大多数场景）",
    },
    "deep": {
        "overview_interval": 2.0,
        "overview_layout": "5x5",
        "overview_cap": 50,
        "second_cap": 150,
        "stt_default": True,
        "max_frames": 200,
        "originals": 6,
        "desc": "总览 + 关键段秒图，最多 200 帧，STT + 时间戳，重要帧另发原图（精度优先）",
    },
    "long-video": {
        "overview_interval": 5.0,
        "overview_layout": "4x4",
        "overview_cap": 32,
        "second_cap": 96,
        "stt_default": True,
        "max_frames": 128,
        "originals": 0,
        "desc": ">10 分钟：分段 map-reduce，先分段摘要再汇总",
    },
    "custom-range": {
        "overview_interval": 2.0,
        "overview_layout": "4x4",
        "overview_cap": 16,
        "second_cap": 96,
        "stt_default": True,
        "max_frames": 112,
        "originals": 0,
        "desc": "只处理用户指定时间段并生成秒图",
    },
}

AUTO_LONG_VIDEO_THRESHOLD = 600.0  # 秒；>= 10 分钟则 auto 走 long-video


def resolve_mode(mode: str, duration: float) -> str:
    """auto 模式下按时长选择模式：>= 10 分钟走 long-video，否则 standard。"""
    if mode != "auto":
        return mode
    if duration >= AUTO_LONG_VIDEO_THRESHOLD:
        return "long-video"
    return "standard"


def parse_range(text: Optional[str]) -> Optional[Tuple[float, float]]:
    """解析 '120-180' / '00:02:00-00:03:00' 形式的时间段。"""
    if not text:
        return None
    text = text.strip().replace("~", "-")
    if "-" not in text:
        raise ValueError(f"--range 需形如 120-180 或 00:02:00-00:03:00，收到：{text}")

    def to_sec(s: str) -> float:
        s = s.strip()
        if not s:
            raise ValueError("空的时间值")
        if ":" in s:
            parts = [float(x) for x in s.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0.0)
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return float(s)

    a, b = text.split("-", 1)
    start, end = to_sec(a), to_sec(b)
    if end <= start:
        raise ValueError(f"时间段结束必须大于开始：{text}")
    return (start, end)


# --------------------------------------------------------------------------- #
# 尺寸预测（用于「不真正拼图」就估算 token）
# --------------------------------------------------------------------------- #
def predict_sheet_size(
    src_w: int, src_h: int, layout: str, cell: int,
    has_title: bool = True, cells: Optional[int] = None,
) -> Tuple[int, int]:
    """按源视频宽高比预测拼图成品尺寸，避免为了估 token 而真的拼图。

    `cells` 用于「最后一张不满」的情况：拼图会收缩到实际用到的行数，
    这里跟着算，token 估算才不会被高估。
    """
    cols, rows = parse_layout(layout)
    if cells is not None:
        rows = max(1, math.ceil(max(1, int(cells)) / cols))
    cw, ch = _cell_size(int(src_w or 1280), int(src_h or 720), cell)
    block_w, block_h = cw + PAD, ch + LABEL_BAR + PAD
    title_h = 34 if has_title else 0
    return cols * block_w + PAD, rows * block_h + PAD + title_h


def _fake_sheet(w: int, h: int, layout: str, cells: int) -> Dict:
    """构造一个只有尺寸、没有真实文件的 sheet 字典，专供 token 预估。"""
    return {"sheet_width": w, "sheet_height": h, "layout": layout, "cells": cells}


# --------------------------------------------------------------------------- #
# 预算规划
# --------------------------------------------------------------------------- #
def plan_sampling(
    meta: Dict,
    mode: str,
    profile: Dict[str, Any],
    max_frames: int,
    max_tokens: int,
    cell: int,
    second_frames: int,
    custom_range: Optional[Tuple[float, float]],
    model_preset: Optional[str],
    prompt_text: str,
    transcript_segments: Optional[List[Dict]] = None,
    allow_degrade: bool = True,
    overview_cap: Optional[int] = None,
) -> Dict:
    """在「不抽任何帧」的前提下，先算出采样计划与 token 估算，必要时降级。

    Returns:
        {
          "ok": bool,
          "overview_interval": float,
          "overview_layout": str,
          "overview_frame_count": int,
          "overview_sheet_count": int,
          "second_seconds": [float,...],
          "second_sheet_count": int,
          "frames_total": int,
          "estimate": {...},
          "budget": {...},
          "warnings": [...],
          "degraded": bool,
        }
    """
    src_w, src_h = int(meta.get("width") or 1280), int(meta.get("height") or 720)
    duration = float(meta.get("duration") or 0.0)

    # ---- 处理区间 ----
    if custom_range:
        lo, hi = max(0.0, custom_range[0]), min(duration, custom_range[1])
    else:
        lo, hi = 0.0, duration
    span = max(0.0, hi - lo)

    ov_layout = profile["overview_layout"]
    ov_cols, ov_rows = parse_layout(ov_layout)
    per_overview_sheet = ov_cols * ov_rows

    warnings: List[str] = []
    degraded = False

    # ---- 总览层：由「预算上限」和「建议间隔」共同决定间隔 ----
    interval = float(profile["overview_interval"])
    # 总览帧数上限：优先用调用方显式指定的 overview_cap，否则用模式默认值
    ov_cap = int(overview_cap) if overview_cap else int(profile["overview_cap"])
    if profile["second_cap"] > 0:
        # 有秒图层：总览最多占一半帧预算，另一半留给秒图
        ov_cap = min(ov_cap, max(per_overview_sheet, max_frames // 2))
    else:
        # 无秒图层（quick）：总览可以吃掉全部帧预算
        ov_cap = min(ov_cap, max(per_overview_sheet, max_frames))

    if span > 0:
        # 若按建议间隔抽帧会超过 cap，则自动放宽间隔
        needed = math.ceil(span / interval)
        if needed > ov_cap:
            interval = span / ov_cap
            degraded = True
            warnings.append(
                f"总览帧数按 {profile['overview_interval']:g}s 间隔会达到 {needed} 帧，"
                f"超过总览预算 {ov_cap} 帧，已自动把总览间隔放宽到 {interval:.2f}s。"
            )
        ov_count = min(math.ceil(span / interval) if interval > 0 else 0, ov_cap)
    else:
        ov_count = 0

    ov_sheet_count = math.ceil(ov_count / per_overview_sheet) if per_overview_sheet else 0

    # ---- 秒图层：预算扣除总览层后剩下的 ----
    second_budget_frames = max(0, min(int(profile["second_cap"]), max_frames - ov_count))
    second_seconds: List[float] = []

    if second_budget_frames >= second_frames and profile["second_cap"] > 0:
        max_second_sheets = second_budget_frames // max(1, second_frames)

        scene_changes: List[float] = []
        speech_starts: List[float] = []

        if mode == "custom-range" and custom_range:
            # 指定区间：逐秒铺满
            step = 1.0
            t = lo
            while t < hi - 1e-6 and len(second_seconds) < max_second_sheets:
                second_seconds.append(round(t, 3))
                t += step
        else:
            scene_changes = detect_scene_changes(
                meta["path"], threshold=0.30, start=lo, duration=span if span > 0 else None
            )
            if transcript_segments:
                speech_starts = speech_active_seconds(transcript_segments)
            second_seconds = pick_key_seconds(
                duration=duration,
                budget_frames=max_second_sheets * second_frames,
                frames_per_second_sheet=second_frames,
                scene_changes=scene_changes,
                speech_starts=speech_starts,
                user_range=(lo, hi),
                min_gap=1.0,
            )

    # ---- 预估 token ----
    sec_layout = "3x2"
    sec_w, sec_h = predict_sheet_size(src_w, src_h, sec_layout, cell, cells=second_frames)

    # 总览图逐张预测：最后一张可能不满，尺寸会缩小
    ov_fake: List[Dict] = []
    for i in range(ov_sheet_count):
        left = ov_count - i * per_overview_sheet
        n_cells = max(1, min(per_overview_sheet, left))
        w, h = predict_sheet_size(src_w, src_h, ov_layout, cell, cells=n_cells)
        ov_fake.append(_fake_sheet(w, h, ov_layout, n_cells))
    sec_fake = [_fake_sheet(sec_w, sec_h, sec_layout, second_frames) for _ in second_seconds]

    # 转写文本 token 也要算进去（按已压缩后的实际内容）
    transcript_text = " ".join(str(s.get("text", "")) for s in (transcript_segments or []))
    est = estimate_pack(
        overview_sheets=ov_fake,
        second_sheets=sec_fake,
        prompt=prompt_text,
        transcript=[{"text": transcript_text}] if transcript_text else [],
        model=model_preset,
    )

    frames_total = ov_count + len(second_seconds) * second_frames

    # ---- 超限降级：优先砍秒图层（从尾部砍，保留靠前的关键段） ----
    if allow_degrade:
        sheets_before = len(second_seconds)
        guard = 0
        while est["total"] > max_tokens and len(second_seconds) > 0 and guard < 60:
            cut = max(1, len(second_seconds) // 5)
            second_seconds = second_seconds[: max(0, len(second_seconds) - cut)]
            sec_fake = [_fake_sheet(sec_w, sec_h, sec_layout, second_frames) for _ in second_seconds]
            est = estimate_pack(ov_fake, sec_fake, prompt_text,
                                [{"text": transcript_text}] if transcript_text else [],
                                model=model_preset)
            frames_total = ov_count + len(second_seconds) * second_frames
            degraded = True
            guard += 1
        if len(second_seconds) < sheets_before:
            warnings.append(
                f"为把 token 压到上限 {max_tokens} 以内，秒图数量已从 {sheets_before} 张自动下调到 "
                f"{len(second_seconds)} 张。如需更多细节：提高 --max-tokens，或用 --range 聚焦某段时间。"
            )
        if est["total"] > max_tokens:
            warnings.append(
                f"即使已把秒图降到最低（{len(second_seconds)} 张），估算 token {est['total']} 仍超过上限 "
                f"{max_tokens}。此时建议：改用 quick 模式、用 --range 只处理某段时间，"
                f"或调小 --cell / 调大 --overview-interval 以降低图像 token。"
            )

    budget = budget_check(est, max_tokens, frames_total, max_frames)
    if budget["over_budget"]:
        warnings.append(budget["message"])
        warnings.append("可选降级方案：" + "；".join(budget["suggestions"][:4]))
    else:
        warnings.append(budget["message"])

    return {
        "ok": not budget["over_budget"],
        "range": [lo, hi],
        "overview_interval": round(interval, 3),
        "overview_layout": ov_layout,
        "overview_frame_count": ov_count,
        "overview_sheet_count": ov_sheet_count,
        "second_seconds": second_seconds,
        "second_sheet_count": len(second_seconds),
        "second_layout": sec_layout,
        "frames_total": frames_total,
        "estimate": est,
        "budget": budget,
        "warnings": warnings,
        "degraded": degraded,
    }


# --------------------------------------------------------------------------- #
# 主函数
# --------------------------------------------------------------------------- #
def build_video_context(
    video_path: str,
    mode: str = "auto",
    stt: Optional[bool] = None,
    max_tokens: int = 50000,
    custom_range: Any = None,
    out: Optional[str] = None,
    out_dir: Optional[str] = None,
    cell: int = 224,
    second_frames: int = 6,
    second_layout: str = "3x2",
    overview_interval: Optional[float] = None,
    overview_layout: Optional[str] = None,
    overview_cap: Optional[int] = None,
    max_frames: Optional[int] = None,
    model_preset: Optional[str] = None,
    question: Optional[str] = None,
    whisper_model: str = "small",
    whisper_device: str = "auto",
    whisper_compute_type: str = "int8",
    language: Optional[str] = None,
    max_transcript_chars: int = 6000,
    allow_degrade: bool = True,
    keep_temp: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """把视频转换成「视频上下文包」。

    Args:
        video_path: 输入视频
        mode: auto / quick / standard / deep / long-video / custom-range
        stt: None 表示按模式默认（standard/deep 开，quick 关）
        max_tokens: token 上限，超过则提醒并（默认）自动降级
        custom_range: (start, end) 秒，或 '120-180' 字符串；指定后只处理该段
        out: 输出 JSON 路径
        out_dir: 产物（拼图 + JSON）目录，默认取 out 所在目录
        cell: 拼图每小格短边像素（默认 224，不低于 224 以保证可读）
        second_frames: 每张秒图的帧数（默认 6）
        second_layout: 秒图布局（默认 3x2）
        overview_interval / overview_layout / max_frames: 覆盖模式默认值
        overview_cap: 总览层最大采样帧数。想按固定间隔抽更密的总览时要一起调大
            （否则会被模式默认上限自动放宽间隔，达不到你想要的密度）
        model_preset: 目标模型预设（generic/qwen-vl/gpt-4o/claude...），影响 token 估算与提示词
        question: 用户问题，会追加到提示词末尾
        allow_degrade: 超预算时是否自动降级（False 则只提醒）
        keep_temp: 保留抽出的原始帧（排查用）

    Returns:
        输出 JSON 的完整字典（同时按 out 落盘）
    """
    t0 = time.time()
    log = (lambda *a: print(*a, file=sys.stderr)) if progress else (lambda *a: None)

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    # ---------- 1. 探测元数据 ----------
    meta = probe_video(str(video))
    duration = float(meta["duration"])
    log(f"[1/6] 元数据: {duration:.2f}s, {meta['fps']}fps, "
        f"{meta['width']}x{meta['height']}, 音轨={'有' if meta['has_audio'] else '无'}")

    # ---------- 2. 解析模式与参数 ----------
    mode = resolve_mode(mode, duration)
    if mode not in MODE_PROFILES:
        raise ValueError(f"未知模式 '{mode}'，可选：{', '.join(MODE_PROFILES)} 或 auto")
    profile = dict(MODE_PROFILES[mode])
    if overview_interval:
        profile["overview_interval"] = float(overview_interval)
    if overview_layout:
        profile["overview_layout"] = str(overview_layout)
    if max_frames:
        profile_max_frames = int(max_frames)
    else:
        profile_max_frames = int(profile["max_frames"])

    rng = parse_range(custom_range) if isinstance(custom_range, str) else custom_range
    if mode == "custom-range" and not rng:
        raise ValueError("mode=custom-range 必须提供 custom_range，例如 '120-180'")
    if rng and mode != "custom-range":
        # 给了区间但不是 custom-range：自动升级为 custom-range，避免用户困惑
        mode = "custom-range"
        profile = dict(MODE_PROFILES["custom-range"])

    # STT 开关：None -> 按模式默认；无音轨强制关
    want_stt = profile["stt_default"] if stt is None else bool(stt)
    if not meta["has_audio"]:
        want_stt = False

    log(f"[2/6] 模式: {mode} —— {profile['desc']}；STT={'开' if want_stt else '关'}")

    # ---------- 3. 输出路径 ----------
    if out:
        out_path = Path(out).resolve()
    else:
        out_path = (video.parent / f"{video.stem}.video_context.json").resolve()
    assets_dir = Path(out_dir).resolve() if out_dir else (out_path.parent / f"{out_path.stem}_assets")
    sheets_dir = assets_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="vc_b_"))
    frames_dir = tmp_root / "frames"
    log(f"[3/6] 产物目录: {assets_dir}")

    # ---------- 4. STT（先做，因为语音活动会影响关键秒选择） ----------
    transcript_info: Dict[str, Any] = {
        "status": "skipped", "reason": "未启用 STT", "segments": [], "text": "",
        "char_count": 0, "truncated": False, "language": None,
    }
    transcript_segments: List[Dict] = []
    if want_stt:
        log(f"[4/6] STT 转写中（faster-whisper / {whisper_model}）…")
        transcript_info = transcribe(
            str(video),
            model_size=whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
            language=language,
            has_audio=bool(meta["has_audio"]),
            max_chars=max_transcript_chars,
        )
        transcript_segments = transcript_info.get("segments", []) or []
        log(f"      -> {transcript_info['status']} ({len(transcript_segments)} 段)"
            + (f" | {transcript_info['reason']}" if transcript_info["reason"] else ""))
    else:
        log(f"[4/6] STT 已跳过（{transcript_info['reason']}）")

    # ---------- 5. 采样规划 + token 预算 ----------
    draft_prompt = build_prompt(
        meta=meta, mode=mode, overview_sheets=[], second_sheets=[],
        transcript=transcript_segments, custom_range=list(rng) if rng else None,
        user_question=question,
    )
    plan = plan_sampling(
        meta=meta, mode=mode, profile=profile, max_frames=profile_max_frames,
        max_tokens=max_tokens, cell=cell, second_frames=second_frames,
        custom_range=rng, model_preset=model_preset, prompt_text=draft_prompt,
        transcript_segments=transcript_segments, allow_degrade=allow_degrade,
        overview_cap=overview_cap,
    )
    log(f"[5/6] 采样计划: 总览 {plan['overview_frame_count']} 帧 / {plan['overview_sheet_count']} 张"
        f"（间隔 {plan['overview_interval']:g}s）+ 秒图 {plan['second_sheet_count']} 张"
        f"（{second_frames} 帧/张）")
    log(f"      预估 token: 图像 {plan['estimate']['images']} + 文本 {plan['estimate']['text']}"
        f" = {plan['estimate']['total']} / 上限 {max_tokens}")
    for w in plan["warnings"]:
        log(f"      ! {w}")

    # ---------- 6. 抽帧 + 拼图 ----------
    log("[6/6] 抽帧与拼图中…")
    lo, hi = plan["range"]

    # 6.1 总览图
    overview_sheets: List[Dict] = []
    ov_dir = frames_dir / "overview"
    remaining = plan["overview_frame_count"]
    if remaining > 0:
        frames = extract_interval_frames(
            str(video), str(ov_dir),
            interval=plan["overview_interval"], src_fps=float(meta["fps"] or 25.0),
            max_width=960, prefix="ov",
            start=lo, end=hi if hi > lo else None, limit=remaining,
        )
        cols, rows = parse_layout(plan["overview_layout"])
        per = cols * rows
        for i in range(0, len(frames), per):
            chunk = frames[i:i + per]
            if not chunk:
                continue
            sp = sheets_dir / f"overview_{i // per:02d}_{int(chunk[0][0])}s.jpg"
            title = (f"OVERVIEW #{i // per + 1}  "
                     f"{format_timestamp(chunk[0][0])} - {format_timestamp(chunk[-1][0])}  "
                     f"({plan['overview_interval']:g}s/sample)")
            info = build_sheet(
                chunk, str(sp), layout=plan["overview_layout"], cell=cell, title=title,
            )
            info["interval"] = plan["overview_interval"]
            info["frame_timestamps"] = [round(t, 3) for t, _ in chunk]
            overview_sheets.append(info)
        log(f"      -> 总览图 {len(overview_sheets)} 张（{len(frames)} 帧）")

    # 6.2 秒图
    second_sheets: List[Dict] = []
    if plan["second_seconds"]:
        sec_dir = frames_dir / "second"
        second_sheets = build_second_sheets(
            str(video), str(sec_dir), plan["second_seconds"],
            frames_per_sheet=second_frames, window=1.0,
            layout=second_layout, cell=cell, src_fps=float(meta["fps"] or 25.0),
            max_width=960,
        )
        # 把最终文件挪到产物目录（build_second_sheets 落在临时目录）
        for s in second_sheets:
            src = Path(s["path"])
            dst = sheets_dir / src.name
            try:
                if src.resolve() != dst.resolve():
                    dst.write_bytes(src.read_bytes())
                    s["path"] = str(dst)
            except Exception:
                s["path"] = str(src)
        log(f"      -> 秒图 {len(second_sheets)} 张")

    # 6.3 deep：重要帧原图
    original_frames: List[Dict] = []
    n_orig = int(profile.get("originals") or 0)
    if n_orig and plan["second_seconds"]:
        for sec in plan["second_seconds"][:n_orig]:
            p = assets_dir / f"original_{int(sec)}s.jpg"
            try:
                extract_single_frame(str(video), str(p), sec,
                                     src_fps=float(meta["fps"] or 25.0), max_width=1280)
                original_frames.append({
                    "path": str(p), "timestamp": round(float(sec), 3),
                    "timestamp_label": format_timestamp(sec),
                    "width": int(meta["width"]), "height": int(meta["height"]),
                })
            except Exception:
                pass
        if original_frames:
            log(f"      -> 重要帧原图 {len(original_frames)} 张")

    # ---------- 7. 提示词 + 最终 token 估算 ----------
    coverage = (plan["frames_total"] / (duration * float(meta.get("fps") or 1) or 1))
    token_note = (
        f"本片共 {duration:.1f}s，为控制 token，只采样了 {plan['frames_total']} 帧"
        f"（总览每 {plan['overview_interval']:g}s 一帧；秒图共 {len(second_sheets)} 张，每张覆盖 1 秒）。"
        f"抽样帧之外的时间段没有被观测，不要凭想象补全。"
    )

    prompt = build_prompt(
        meta=meta, mode=mode,
        overview_sheets=overview_sheets, second_sheets=second_sheets,
        transcript=transcript_segments, original_frames=original_frames,
        user_question=question, custom_range=list(rng) if rng else None,
        token_note=token_note,
    )

    estimate = estimate_pack(
        overview_sheets=overview_sheets, second_sheets=second_sheets,
        prompt=prompt, transcript=transcript_segments,
        original_frames=original_frames, model=model_preset,
    )
    budget = budget_check(estimate, max_tokens, plan["frames_total"], profile_max_frames)

    # ---------- 8. long-video：分段 map-reduce 计划 ----------
    map_reduce: Optional[Dict] = None
    if mode == "long-video":
        chunk_sec = 120.0
        chunks = []
        idx = 1
        t = lo
        while t < hi:
            e = min(hi, t + chunk_sec)
            cmap = build_map_prompt(meta, idx, math.ceil((hi - lo) / chunk_sec), t, e)
            c_overview = [
                s for s in overview_sheets if s.get("start", 0) >= t - 1e-6 and s.get("start", 0) < e
            ]
            c_second = [
                s for s in second_sheets if s.get("start", 0) >= t - 1e-6 and s.get("start", 0) < e
            ]
            chunks.append({
                "index": idx, "start": round(t, 3), "end": round(e, 3),
                "start_label": format_timestamp(t), "end_label": format_timestamp(e),
                "overview_sheets": c_overview, "second_sheets": c_second,
                "map_prompt": cmap,
            })
            idx += 1
            t = e
        map_reduce = {
            "chunk_count": len(chunks),
            "chunk_seconds": chunk_sec,
            "chunks": chunks,
            "reduce_prompt_template": build_reduce_prompt(meta, ["<第 i 段摘要>"], question),
            "how_to": (
                "对每个 chunk：把 chunk.overview_sheets + chunk.second_sheets 作为图片、"
                "chunk.map_prompt 作为文本，发给目标模型得到该段摘要；"
                "再把所有段摘要填入 reduce_prompt_template 的占位处，做一次汇总回答。"
            ),
        }
        log(f"      -> long-video 分段计划：{len(chunks)} 段（每段 {chunk_sec:g}s）")

    # ---------- 9. 组装输出 ----------
    result: Dict[str, Any] = {
        "meta": {
            "duration": round(duration, 3),
            "fps": round(float(meta["fps"]), 3),
            "has_audio": bool(meta["has_audio"]),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "nb_frames": int(meta.get("nb_frames") or 0),
            "video_codec": meta.get("video_codec", ""),
            "audio_codec": meta.get("audio_codec", ""),
            "path": meta["path"],
        },
        "mode": mode,
        "range": plan["range"],
        "custom_range": list(rng) if rng else None,
        "overview_sheets": [
            {k: v for k, v in s.items() if k != "frame_timestamps"} for s in overview_sheets
        ],
        "second_sheets": second_sheets,
        "original_frames": original_frames,
        "transcript": transcript_segments,
        "transcript_info": {
            "status": transcript_info.get("status"),
            "reason": transcript_info.get("reason"),
            "language": transcript_info.get("language"),
            "truncated": transcript_info.get("truncated"),
            "char_count": transcript_info.get("char_count"),
        },
        "prompt": prompt,
        "token_estimate": {
            "images": estimate["images"],
            "text": estimate["text"],
            "total": estimate["total"],
        },
        "token_breakdown": estimate["breakdown"],
        "budget": {
            "max_tokens": max_tokens,
            "over_budget": budget["over_budget"],
            "ratio": budget["ratio"],
            "message": budget["message"],
        },
        "sampling_plan": {
            "overview_interval": plan["overview_interval"],
            "overview_layout": plan["overview_layout"],
            "overview_frame_count": plan["overview_frame_count"],
            "second_layout": plan["second_layout"],
            "second_frames_per_sheet": second_frames,
            "second_sheet_count": plan["second_sheet_count"],
            "frames_total": plan["frames_total"],
            "max_frames": profile_max_frames,
            "degraded": plan["degraded"],
        },
        "warnings": plan["warnings"],
        "assets_dir": str(assets_dir),
        "model_preset": model_preset or "generic",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if map_reduce is not None:
        result["map_reduce"] = map_reduce

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["json_path"] = str(out_path)

    # ---------- 10. 清理临时文件 ----------
    if keep_temp:
        log(f"临时帧目录已保留：{tmp_root}")
    else:
        cleanup_dir(str(tmp_root))

    log(f"完成，用时 {result['elapsed_sec']}s -> {out_path}")
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="build_video_context",
        description="把视频转换成「多模态模型可消费的视频上下文包」（总览图 + 关键秒图 + 可选转写 + 提示词 + token 估算）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 自动模式（<=10 分钟走 standard，>10 分钟走 long-video）
  python scripts/build_video_context.py --video in.mp4 --mode auto --stt --max-tokens 50000 --out out.json

  # 只看 2:00-3:00，并逐秒出秒图
  python scripts/build_video_context.py --video in.mp4 --mode custom-range --range 120-180 --out seg.json

  # 超省 token：只要总览
  python scripts/build_video_context.py --video in.mp4 --mode quick --max-tokens 8000 --out quick.json

  # 精度优先 + 指定目标模型（影响 token 估算）
  python scripts/build_video_context.py --video in.mp4 --mode deep --model-preset qwen-vl --out deep.json
""",
    )
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "quick", "standard", "deep", "long-video", "custom-range"],
                    help="处理模式，默认 auto")
    stt_group = ap.add_mutually_exclusive_group()
    stt_group.add_argument("--stt", dest="stt", action="store_true", default=None,
                           help="启用 STT（默认按模式：standard/deep 开，quick 关）")
    stt_group.add_argument("--no-stt", dest="stt", action="store_false",
                           help="关闭 STT")
    ap.add_argument("--max-tokens", type=int, default=50000, help="token 上限，默认 50000")
    ap.add_argument("--range", dest="custom_range", default=None,
                    help="只处理该时间段，如 120-180 或 00:02:00-00:03:00")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--out-dir", default=None, help="拼图与产物目录")
    ap.add_argument("--cell", type=int, default=224, help="每小格短边像素，默认 224")
    ap.add_argument("--second-frames", type=int, default=6, help="每张秒图帧数，默认 6")
    ap.add_argument("--second-layout", default="3x2", help="秒图布局，默认 3x2")
    ap.add_argument("--overview-interval", type=float, default=None, help="总览抽帧间隔（秒）")
    ap.add_argument("--overview-layout", default=None, help="总览布局，如 4x4 / 5x5")
    ap.add_argument("--overview-cap", type=int, default=None,
                    help="总览层最大采样帧数。想按固定间隔抽更密的总览时调大它"
                         "（例如 62 秒视频想每 1 秒 1 帧 → --overview-interval 1 --overview-cap 70）")
    ap.add_argument("--max-frames", type=int, default=None, help="总帧数上限（覆盖模式默认值）")
    ap.add_argument("--model-preset", default=None,
                    help="目标模型预设（generic/qwen-vl/gpt-4o/claude 等），影响 token 估算")
    ap.add_argument("--question", default=None, help="用户问题，追加到提示词末尾")
    ap.add_argument("--whisper-model", default="small", help="faster-whisper 模型大小，默认 small")
    ap.add_argument("--whisper-device", default="auto", help="cpu / cuda / auto")
    ap.add_argument("--whisper-compute-type", default="int8", help="int8 / float16 / int8_float16")
    ap.add_argument("--language", default=None, help="转写语言，如 zh / en")
    ap.add_argument("--max-transcript-chars", type=int, default=6000, help="转写文本字数上限")
    ap.add_argument("--no-degrade", dest="allow_degrade", action="store_false", default=True,
                    help="超预算时不自动降级，只提醒")
    ap.add_argument("--keep-temp", action="store_true", help="保留临时抽帧目录（排查用）")
    ap.add_argument("--quiet", action="store_true", help="不打印进度")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = build_video_context(
            video_path=args.video,
            mode=args.mode,
            stt=args.stt,
            max_tokens=args.max_tokens,
            custom_range=args.custom_range,
            out=args.out,
            out_dir=args.out_dir,
            cell=args.cell,
            second_frames=args.second_frames,
            second_layout=args.second_layout,
            overview_interval=args.overview_interval,
            overview_layout=args.overview_layout,
            overview_cap=args.overview_cap,
            max_frames=args.max_frames,
            model_preset=args.model_preset,
            question=args.question,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            whisper_compute_type=args.whisper_compute_type,
            language=args.language,
            max_transcript_chars=args.max_transcript_chars,
            allow_degrade=args.allow_degrade,
            keep_temp=args.keep_temp,
            progress=not args.quiet,
        )
    except FFmpegNotFoundError as exc:
        print(f"\n[环境错误] {exc}\n", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\n[失败] {type(exc).__name__}: {exc}\n", file=sys.stderr)
        return 1

    # stdout 打印「人类可读摘要」，便于 CLI 直接看结果
    print(json.dumps({
        "json": result.get("json_path"),
        "mode": result["mode"],
        "meta": result["meta"],
        "overview_sheets": len(result["overview_sheets"]),
        "second_sheets": len(result["second_sheets"]),
        "transcript_segments": len(result["transcript"]),
        "token_estimate": result["token_estimate"],
        "over_budget": result["budget"]["over_budget"],
        "warnings": result["warnings"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
