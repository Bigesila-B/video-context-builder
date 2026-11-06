"""Duration-aware sampling with a shared input budget and temporal coverage.

Pure planning functions: no decoding, model requests, or model downloads.
The estimates are heuristics; the caller must reconcile the rendered payload.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence

from make_contact_sheet import LABEL_BAR, PAD, _cell_size, parse_layout
from token_estimator import budget_check, estimate_image_tokens, estimate_text_tokens


def duration_policy(span: float) -> Dict:
    """Nominal density is a target, not a promise when the budget is small."""
    if not math.isfinite(span) or span <= 0:
        raise ValueError("视频/区间时长必须为有限正数")
    interval = next(i for end, i in (
        (60, 0.5), (300, 1.0), (1200, 2.0), (3600, 4.0), (math.inf, 8.0)
    ) if span <= end)
    tokens = min(40000, max(6000, math.ceil((4000 + 2200 * math.sqrt(span / 30)) / 1000) * 1000))
    return {"target_interval": interval, "suggested_max_tokens": tokens,
            "overview_layout": "3x2", "detail_window": 2.0,
            "max_frames": 2048}


def token_allocation(max_tokens: int, response_tokens: Optional[int] = None,
                     safety_ratio: float = 0.10, revisit_ratio: float = 0.15) -> Dict:
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("max_tokens 必须为正整数")
    if not all(math.isfinite(x) and 0 <= x < 1 for x in (safety_ratio, revisit_ratio)):
        raise ValueError("安全余量和补看比例必须在 [0, 1) 内")
    response = min(2048, max(256, max_tokens // 10)) if response_tokens is None else response_tokens
    if not isinstance(response, int) or isinstance(response, bool) or response < 0:
        raise ValueError("response_tokens 必须为非负整数")
    safety, revisit = math.ceil(max_tokens * safety_ratio), math.ceil(max_tokens * revisit_ratio)
    limit = max_tokens - response - safety - revisit
    if limit <= 0:
        raise ValueError("预算不足以容纳回答、安全余量和补看预留，请提高 max_tokens 或降低预留")
    return {"max_tokens": max_tokens, "input_limit": limit,
            "response_reserved": response, "safety_reserved": safety,
            "revisit_reserved": revisit, "scope": "single_request"}


def evenly_spaced(items: Sequence, count: int) -> List:
    """Retain both ends when possible; never truncate only the tail."""
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def question_terms(question: Optional[str]) -> List[str]:
    text = (question or "").lower()
    # Lexical hints only: no claim of semantic retrieval.
    for word in ("视频", "什么", "怎么", "里面", "这个", "发生", "时候", "请问", "哪个"):
        text = text.replace(word, " ")
    terms = set(re.findall(r"[a-z0-9]{3,}", text)) - {"the", "and", "what", "when", "video"}
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
    return sorted(terms)


def relevance(text: str, terms: Sequence[str]) -> int:
    return sum(term in text.lower() for term in terms)


def select_transcript(segments: Sequence[Dict], token_limit: int, question: Optional[str],
                      max_chars: int = 6000) -> List[Dict]:
    """Select verbatim segments, preserving original timestamps and full source elsewhere."""
    if token_limit <= 0 or max_chars <= 0 or not segments:
        return []
    costs = [estimate_text_tokens(f"[{s['start']:.3f}-{s['end']:.3f}] {s['text']}\n") for s in segments]
    terms = question_terms(question)
    # Half the text budget protects timeline coverage; the rest prefers query matches.
    selected, used, chars = set(), 0, 0
    for i in evenly_spaced(list(range(len(segments))), min(40, len(segments))):
        n = len(segments[i]["text"])
        if used + costs[i] <= token_limit // 2 and chars + n <= max_chars:
            selected.add(i)
            used += costs[i]
            chars += n
    ranked = sorted(range(len(segments)), key=lambda i: (-relevance(segments[i]["text"], terms), i))
    # Equal-score segments are visited in temporal spread order, not prefix order.
    scores = sorted({relevance(segments[i]["text"], terms) for i in ranked}, reverse=True)
    for score in scores:
        group = [i for i in ranked if relevance(segments[i]["text"], terms) == score and i not in selected]
        # Recursively interleave halves to distribute a tight budget over the full span.
        order = spread_order(group)
        for i in order:
            n = len(segments[i]["text"])
            if len(selected) < 400 and used + costs[i] <= token_limit and chars + n <= max_chars:
                selected.add(i)
                used += costs[i]
                chars += n
    return [dict(segments[i]) for i in sorted(selected)]


def spread_order(items: Sequence) -> List:
    if len(items) <= 2:
        return list(items)
    # First/last, then progressively fill the largest gaps.
    result, pending = [items[0], items[-1]], [(1, len(items) - 1)]
    while pending:
        a, b = pending.pop(0)
        if a >= b:
            continue
        m = (a + b) // 2
        result.append(items[m])
        pending.extend(((a, m), (m + 1, b)))
    return result


def detail_candidates(lo: float, hi: float, scene_changes: Sequence[float],
                      transcript: Sequence[Dict], question: Optional[str], window: float) -> List[Dict]:
    terms = question_terms(question)
    candidates: Dict[float, Dict] = {}

    def add(t: float, score: float, source: str) -> None:
        if not math.isfinite(t) or not lo <= t < hi:
            return
        start = round(max(lo, min(t - window / 2, hi - window)), 3)
        if start not in candidates or candidates[start]["score"] < score:
            candidates[start] = {"start": start, "score": score, "source": source}

    for t in scene_changes:
        add(t, 3, "scene_change")
    for seg in transcript:
        if seg.get("timestamp_source") == "chunk":
            continue  # Chunk edges are not evidence of speech/event onset.
        matches = relevance(seg.get("text", ""), terms)
        add(float(seg.get("start", lo)), 1 + min(8, matches * 3),
            "question_keyword" if matches else "speech")
    # Bounded uniform fallbacks provide coverage even when detectors find nothing.
    n = min(512, max(1, math.ceil((hi - lo) / max(window, 10))))
    for i in range(n):
        add(lo + (i + 0.5) * (hi - lo) / n, 0.5, "uniform")
    return sorted(candidates.values(), key=lambda c: c["start"])


def choose_details(candidates: Sequence[Dict], count: int, lo: float, hi: float,
                   window: float) -> List[Dict]:
    """One choice per time stratum, favoring local changes and query matches."""
    if count <= 0:
        return []
    chosen = []
    for i in range(count):
        a, b = lo + i * (hi - lo) / count, lo + (i + 1) * (hi - lo) / count
        group = [c for c in candidates if a <= c["start"] < b]
        if not group:
            continue
        best = max(group, key=lambda c: (c["score"], -abs(c["start"] - (a + b) / 2)))
        if all(abs(best["start"] - c["start"]) >= min(window, 1.0) for c in chosen):
            chosen.append(dict(best))
    return sorted(chosen, key=lambda c: c["start"])


def sheet_tokens(meta: Dict, layout: str, cell: int, count: int,
                 model: Optional[str]) -> int:
    cols, rows = parse_layout(layout)
    cw, ch = _cell_size(int(meta["width"]), int(meta["height"]), cell)
    total = 0
    for start in range(0, count, cols * rows):
        n = min(cols * rows, count - start)
        w = cols * (cw + PAD) + PAD
        h = math.ceil(n / cols) * (ch + LABEL_BAR + PAD) + PAD + 34
        total += estimate_image_tokens(w, h, model) + 80  # manifest/timestamp allowance
    return total


def fit_count(target: int, budget: int, cost) -> int:
    low, high = 0, max(0, target)
    while low < high:
        middle = (low + high + 1) // 2
        if cost(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def plan_adaptive_sampling(meta: Dict, max_tokens: int, allocation: Dict,
                           prompt_text: str, transcript: Sequence[Dict] = (),
                           scene_changes: Sequence[float] = (), question: Optional[str] = None,
                           custom_range=None, cell: int = 224, second_frames: int = 6,
                           second_layout: str = "3x2", model_preset: Optional[str] = None,
                           max_frames: Optional[int] = None, overview_cap: Optional[int] = None,
                           overview_interval: Optional[float] = None,
                           overview_layout: Optional[str] = None, allow_degrade: bool = True) -> Dict:
    lo, hi = custom_range if custom_range else (0.0, float(meta["duration"]))
    policy = duration_policy(hi - lo)
    frame_limit = max_frames if max_frames is not None else policy["max_frames"]
    layout = overview_layout or policy["overview_layout"]
    parse_layout(layout)
    fps = float(meta.get("fps") or 25)
    interval = max(overview_interval or policy["target_interval"], 1 / fps)
    target = min(frame_limit, overview_cap if overview_cap is not None else frame_limit,
                 math.ceil((hi - lo) / interval))
    cost = lambda n: sheet_tokens(meta, layout, cell, n, model_preset)
    text = estimate_text_tokens(prompt_text, model_preset)
    image_budget = max(0, allocation["input_limit"] - text - 400)
    ov_count = fit_count(target, int(image_budget * 0.65), cost) if allow_degrade else target
    window = min(policy["detail_window"], hi - lo)
    candidates = detail_candidates(lo, hi, scene_changes, transcript, question, window)
    detail_cost = sheet_tokens(meta, second_layout, cell, second_frames, model_preset)
    desired_details = min(48, max(1, math.ceil((hi - lo) / 20)), len(candidates))
    detail_count = min(desired_details, max(0, (frame_limit - ov_count) // second_frames))
    if allow_degrade:
        detail_count = min(detail_count, max(0, (image_budget - cost(ov_count)) // max(1, detail_cost)))
    details = choose_details(candidates, detail_count, lo, hi, window)
    # Spend unused detail allowance on missing baseline coverage, while respecting target density.
    if allow_degrade:
        ov_count = fit_count(min(target, frame_limit - len(details) * second_frames),
                             max(0, image_budget - len(details) * detail_cost), cost)
    if ov_count:
        interval = max(interval, (hi - lo) / ov_count)
        # ceil avoids producing extra frames then truncating the end of the timeline.
        interval = math.ceil(interval * fps - 1e-9) / fps
    frames = ov_count + len(details) * second_frames
    images = cost(ov_count) + len(details) * detail_cost
    estimate = {"images": images, "text": text, "total": images + text}
    warnings = []
    if ov_count < target:
        warnings.append(f"按时长的目标总览为 {target} 帧，预算/帧上限下采用 {ov_count} 帧，保留全程均匀覆盖。")
    if not ov_count:
        warnings.append("预算无法容纳总览图，缺少全程画面证据；请提高预算或缩小区间。")
    warnings.append("图像 token 为模型预设的近似值；预留余量不保证实际计费误差上限。")
    return {"ok": estimate["total"] <= allocation["input_limit"],
            "strategy": "adaptive", "range": [lo, hi],
            "overview_interval": interval, "overview_layout": layout,
            "overview_frame_count": ov_count,
            "overview_sheet_count": math.ceil(ov_count / math.prod(parse_layout(layout))),
            "second_seconds": [c["start"] for c in details], "detail_candidates": details,
            "second_sheet_count": len(details), "second_layout": second_layout,
            "second_window": window, "frames_total": frames, "max_frames": frame_limit,
            "estimate": estimate, "budget": budget_check(estimate, allocation["input_limit"], frames, frame_limit),
            "warnings": warnings, "degraded": ov_count < target or len(details) < desired_details,
            "duration_policy": policy, "allocation": allocation}
