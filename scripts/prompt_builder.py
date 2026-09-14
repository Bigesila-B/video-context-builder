"""
面向目标模型的提示词构造器。

产出约束型提示词：告诉「只有图片理解能力」的模型——
它没看过视频，只有抽样帧拼图 + 秒图 + 音频转写；必须引用时间戳、不确定要说明、不许编造。

用法：
    python scripts/prompt_builder.py --meta meta.json --out prompt.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_contact_sheet import format_timestamp  # noqa: E402

# --------------------------------------------------------------------------- #
# 固定指令模板（用户指定的约束，逐条保留）
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION = """你是一个视频理解助手。
你没有直接看视频，只能看到按时间顺序排列的关键帧拼图、秒图和音频转写。
每张秒图对应视频中的 1 秒，包含多个按时间顺序采样的帧。
阅读顺序：从左到右，从上到下。
请遵守：
1. 只基于提供的材料回答；
2. 引用时间戳，例如 [00:12:30]；
3. 不确定时说明“抽样帧中未看到”；
4. 区分画面信息和音频信息；
5. 不要编造未出现的情节。"""


def _fmt_range(a: float, b: float) -> str:
    return f"{format_timestamp(a)}-{format_timestamp(b)}"


def build_material_manifest(
    overview_sheets: List[Dict],
    second_sheets: List[Dict],
    original_frames: Optional[List[Dict]] = None,
    transcript: Optional[List[Dict]] = None,
) -> str:
    """把「有哪些材料、每张图对应哪一秒」写清楚，方便模型对齐图与时间。"""
    lines: List[str] = []

    if overview_sheets:
        lines.append("【总览图 overview_sheets】用于定位全局，每格是一个抽样帧：")
        for i, s in enumerate(overview_sheets, 1):
            lines.append(
                f"  - 图{i}: {_fmt_range(s.get('start', 0), s.get('end', 0))} "
                f"布局 {s.get('layout')}，覆盖 {s.get('cells', '?')} 个采样点，"
                f"采样间隔约 {max(1, int(s.get('interval') or 0))} 秒"
            )
        lines.append("")

    if second_sheets:
        lines.append("【秒图 second_sheets】每张图 = 视频中的 1 秒（格内帧从左到右、从上到下按时间序）：")
        for i, s in enumerate(second_sheets, 1):
            ts = s.get("frame_timestamps") or []
            span = ""
            if ts:
                span = f"，帧时间戳 {format_timestamp(ts[0])} → {format_timestamp(ts[-1])}"
            lines.append(
                f"  - 秒图{i}: {s.get('timestamp') or format_timestamp(s.get('start', 0))} "
                f"（{_fmt_range(s.get('start', 0), s.get('end', 0))}），"
                f"{s.get('frames', '?')} 帧 / {s.get('layout')} 布局{span}"
            )
        lines.append("")

    if original_frames:
        lines.append("【原图 original_frames】重要时刻的高清单帧，用于核对细节：")
        for i, f in enumerate(original_frames, 1):
            lines.append(f"  - 原图{i}: {format_timestamp(f.get('timestamp', 0))}")
        lines.append("")

    if transcript:
        lines.append(f"【音频转写 transcript】共 {len(transcript)} 段，带起止时间戳。")
        lines.append("")
    else:
        lines.append("【音频转写 transcript】无（视频无音轨，或本次未启用/不可用）。")
        lines.append("")

    return "\n".join(lines).rstrip()


def build_transcript_block(segments: Optional[List[Dict]], max_lines: int = 400) -> str:
    """渲染带时间戳的转写文本块。"""
    if not segments:
        return "（无音频转写内容）"
    lines = []
    for s in segments[:max_lines]:
        lines.append(
            f"[{format_timestamp(s.get('start', 0))}-{format_timestamp(s.get('end', 0))}] {s.get('text', '')}"
        )
    if len(segments) > max_lines:
        lines.append(f"...（其余 {len(segments) - max_lines} 段已省略）")
    return "\n".join(lines)


def build_prompt(
    meta: Dict,
    mode: str,
    overview_sheets: List[Dict],
    second_sheets: List[Dict],
    transcript: Optional[List[Dict]] = None,
    original_frames: Optional[List[Dict]] = None,
    user_question: Optional[str] = None,
    custom_range: Optional[List[float]] = None,
    token_note: Optional[str] = None,
) -> str:
    """生成最终发给多模态模型的提示词（图片另行按 manifest 顺序附上）。"""
    duration = float(meta.get("duration") or 0)
    parts: List[str] = []

    parts.append(SYSTEM_INSTRUCTION)
    parts.append("")

    # ---- 视频概况 ----
    parts.append("=== 视频概况 ===")
    parts.append(f"- 时长：{duration:.1f} 秒（{format_timestamp(duration)}）")
    parts.append(f"- 分辨率：{meta.get('width', '?')}x{meta.get('height', '?')}，帧率 {meta.get('fps', '?')} fps")
    parts.append(f"- 音轨：{'有' if meta.get('has_audio') else '无'}")
    parts.append(f"- 处理模式：{mode}")
    if custom_range:
        parts.append(f"- 关注区间：{_fmt_range(custom_range[0], custom_range[1])}（仅该区间有秒图）")
    parts.append("")

    # ---- 材料清单 ----
    parts.append("=== 材料清单（请按此顺序阅读附图） ===")
    parts.append(build_material_manifest(overview_sheets, second_sheets, original_frames, transcript))

    # ---- 转写 ----
    parts.append("=== 音频转写（带时间戳） ===")
    parts.append(build_transcript_block(transcript))
    parts.append("")

    # ---- token 说明（可选） ----
    if token_note:
        parts.append("=== 本次抽样说明 ===")
        parts.append(token_note)
        parts.append("")

    # ---- 回答要求 ----
    parts.append("=== 回答要求 ===")
    parts.append(
        "1) 先给结论/要点，再给证据；每条证据后面用 [MM:SS] 标注来自哪一秒。\n"
        "2) 严格区分「画面证据」（你在图上看到的）与「音频证据」（转写里的），分别标注。\n"
        "3) 抽样帧之间可能有跳跃；跳过的区间若被问到，回答「抽样帧中未看到，该区间未被采样」。\n"
        "4) 不推断未提供材料中不存在的人物、台词或情节。\n"
        "5) 如果材料不足以回答，明确说明「材料不足」，并列出你还需要哪个时间段的秒图。"
    )

    if user_question:
        parts.append("")
        parts.append("=== 用户问题 ===")
        parts.append(user_question)

    return "\n".join(parts)


def build_map_prompt(meta: Dict, chunk_index: int, chunk_total: int, start: float, end: float) -> str:
    """long-video 模式：分段摘要（map 阶段）用的提示词。"""
    return (
        f"你是一个视频理解助手。你没有直接看视频，只能看到按时间顺序排列的关键帧拼图和音频转写。\n"
        f"阅读顺序：从左到右，从上到下。\n\n"
        f"=== 本段概况 ===\n"
        f"- 这是第 {chunk_index}/{chunk_total} 段\n"
        f"- 时间段：{_fmt_range(start, end)}（共 {max(0.0, end - start):.1f} 秒）\n"
        f"- 全片时长：{float(meta.get('duration') or 0):.1f} 秒\n\n"
        f"=== 任务 ===\n"
        f"只针对本段的材料，输出结构化摘要：\n"
        f"1) 本段发生了什么（3-5 条，每条带 [MM:SS] 时间戳）；\n"
        f"2) 画面信息 / 音频信息分开写；\n"
        f"3) 关键人物、物体、场景变化；\n"
        f"4) 不确定的地方写「抽样帧中未看到」，不要编造。"
    )


def build_reduce_prompt(meta: Dict, chunk_summaries: List[str], user_question: Optional[str] = None) -> str:
    """long-video 模式：汇总（reduce 阶段）用的提示词。"""
    joined = "\n\n".join(
        f"--- 第 {i + 1} 段摘要 ---\n{s}" for i, s in enumerate(chunk_summaries)
    )
    tail = f"\n\n=== 用户问题 ===\n{user_question}" if user_question else ""
    return (
        "你是一个视频理解助手。下面是对同一段视频**分段抽样**后得到的各段摘要。\n"
        "你没有看原视频，只能基于这些摘要回答。\n\n"
        "请遵守：\n"
        "1) 只基于提供的内容回答；\n"
        "2) 引用时间戳，例如 [00:12:30]；\n"
        "3) 不确定时说明「摘要中未提及」；\n"
        "4) 区分画面信息和音频信息；\n"
        "5) 不要编造未出现的情节；\n"
        "6) 若摘要之间存在时间断裂，明确指出断裂区间。\n\n"
        f"=== 分段摘要 ===\n{joined}\n\n"
        f"=== 任务 ===\n先给全片结论，再按时间线串联各段，最后指出材料覆盖的空白区。{tail}"
    )


def main() -> int:
    """自检入口：从 build_video_context 产出的 JSON 里重建提示词。"""
    ap = argparse.ArgumentParser(description="生成面向目标模型的提示词")
    ap.add_argument("context_json", help="build_video_context 产出的 JSON")
    ap.add_argument("--question", default=None)
    ap.add_argument("--out", default=None, help="写出到文件")
    args = ap.parse_args()

    data = json.loads(Path(args.context_json).read_text(encoding="utf-8"))
    prompt = build_prompt(
        meta=data.get("meta", {}),
        mode=data.get("mode", "standard"),
        overview_sheets=data.get("overview_sheets", []),
        second_sheets=data.get("second_sheets", []),
        transcript=data.get("transcript", []),
        original_frames=data.get("original_frames", []),
        custom_range=data.get("custom_range"),
        user_question=args.question,
    )
    print(prompt)
    if args.out:
        Path(args.out).write_text(prompt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
