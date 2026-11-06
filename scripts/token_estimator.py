"""
Token 估算与预算控制。

职责：
1. 估算每张拼图的图像 token；
2. 估算提示词 / 转写文本的 token；
3. 汇总成 {images, text, total}，与用户给的 --max-tokens 比较，超限时给出提醒与降级建议。

说明：不同厂商的图像 token 计费方式不同，这里用「业界常见近似」：
  - patch 类（GPT-4o/Qwen-VL/GLM-4V/InternVL 等）：按 28x28 像素一个 patch 计 1 token；
  - area 类（Claude 系）：约 宽度*高度/750；
  估算值只用于近似规划与超限提醒，没有经过跨模型校准的固定误差上限。
  实际计费还受供应商缩放/切块/detail 参数影响；不能替代真实 usage。
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional

# 模型预设：key 为预设名，value 为计算方式
MODEL_PRESETS: Dict[str, Dict] = {
    "generic": {"type": "patch", "patch": 28},
    "qwen-vl": {"type": "patch", "patch": 28},
    "qwen2-vl": {"type": "patch", "patch": 28},
    "gpt-4o": {"type": "patch", "patch": 28},
    "gpt-4v": {"type": "patch", "patch": 28},
    "glm-4v": {"type": "patch", "patch": 28},
    "internvl": {"type": "patch", "patch": 28},
    "claude": {"type": "area", "divisor": 750},
    "claude-3": {"type": "area", "divisor": 750},
}

DEFAULT_PRESET = "generic"

# CJK 字符范围（用于文本 token 启发式估算）
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def resolve_preset(model: Optional[str]) -> Dict:
    """把模型名映射到预设；未知模型回退 generic。"""
    if not model:
        return MODEL_PRESETS[DEFAULT_PRESET]
    key = model.strip().lower()
    if key in MODEL_PRESETS:
        return MODEL_PRESETS[key]
    for name, preset in MODEL_PRESETS.items():
        if name in key:
            return preset
    return MODEL_PRESETS[DEFAULT_PRESET]


def estimate_image_tokens(width: int, height: int, model: Optional[str] = None) -> int:
    """估算单张图片的 token 数。"""
    preset = resolve_preset(model)
    w, h = max(1, int(width)), max(1, int(height))
    if preset["type"] == "patch":
        patch = int(preset["patch"])
        return int(math.ceil(w / patch) * math.ceil(h / patch))
    divisor = float(preset["divisor"])
    return max(1, int(math.ceil(w * h / divisor)))


def estimate_text_tokens(text: str, model: Optional[str] = None) -> int:
    """估算文本 token 数。

    优先用 tiktoken（若已安装）；否则用启发式：
      CJK 字符 ≈ 1 token / 1.5 字，其余字符 ≈ 1 token / 4 字符。
    """
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        pass

    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return int(math.ceil(cjk / 1.5 + other / 4.0))


def estimate_sheets(sheets: Iterable[Dict], model: Optional[str] = None) -> int:
    """累加一组拼图的图像 token。

    每张拼图按「整张拼图的分辨率」计一次（因为模型是把它当一张图看的）。
    """
    total = 0
    for s in sheets or []:
        w = int(s.get("sheet_width") or 0)
        h = int(s.get("sheet_height") or 0)
        if w <= 0 or h <= 0:
            # 兜底：未知尺寸时按 1024x1024 估
            w = h = 1024
        total += estimate_image_tokens(w, h, model)
    return total


def estimate_pack(
    overview_sheets: List[Dict],
    second_sheets: List[Dict],
    prompt: str,
    transcript: List[Dict],
    original_frames: Optional[List[Dict]] = None,
    model: Optional[str] = None,
    per_image_overhead: int = 0,
) -> Dict:
    """汇总整个「视频上下文包」的 token 估算。

    Args:
        overview_sheets / second_sheets: 拼图元数据（需含 sheet_width/sheet_height）
        original_frames: deep 模式下单独发送的原图（含 width/height）
        per_image_overhead: 每张图额外的固定开销 token（部分厂商按图计费）
    Returns:
        {"images": N, "text": M, "total": T,
         "breakdown": {"overview": a, "second": b, "originals": c, "prompt": d, "transcript": e}}
    """
    ov = estimate_sheets(overview_sheets, model)
    sec = estimate_sheets(second_sheets, model)

    orig = 0
    for f in original_frames or []:
        orig += estimate_image_tokens(int(f.get("width") or 1024), int(f.get("height") or 1024), model)

    n_images = len(overview_sheets or []) + len(second_sheets or []) + len(original_frames or [])
    images = ov + sec + orig + per_image_overhead * n_images

    t_prompt = estimate_text_tokens(prompt, model)
    t_trans = estimate_text_tokens(
        " ".join(str(s.get("text", "")) for s in (transcript or [])), model
    )
    text = t_prompt + t_trans

    return {
        "images": int(images),
        "text": int(text),
        "total": int(images + text),
        "breakdown": {
            "overview": int(ov),
            "second": int(sec),
            "originals": int(orig),
            "prompt": int(t_prompt),
            "transcript": int(t_trans),
        },
        "image_count": n_images,
        "model": model or DEFAULT_PRESET,
    }


def budget_check(
    estimate: Dict,
    max_tokens: int,
    frames_used: int,
    max_frames: int,
) -> Dict:
    """把估算结果与阈值比对，产出「提醒 + 降级建议」。

    Returns:
        {"over_budget": bool, "ratio": 0.0-1.0+, "message": "...", "suggestions": [...]}
    """
    total = int(estimate.get("total", 0))
    ratio = (total / max_tokens) if max_tokens else 0.0
    over = total > max_tokens

    suggestions: List[str] = []
    if over or ratio > 0.8:
        suggestions = [
            "quick：只用总览图（最多 32 帧），关闭 STT —— 适合「视频大致讲了什么」。",
            "standard：总览 + 关键秒图（最多 96 帧），适合大多数场景。",
            "deep：关键段秒图（最多 200 帧）+ STT，精度最高、token 最多。",
            "custom-range：只处理用户指定的时间段，例如 --range 120-180，token 最省。",
            "long-video：超过 10 分钟建议分段 map-reduce，先摘要再汇总。",
            "调小 --cell（如 224 -> 160）或 --overview-interval（如 3 -> 5）可显著降低图像 token。",
        ]

    if not over:
        msg = (f"token 估算 {total} <= 上限 {max_tokens}（占比 {ratio:.0%}，"
               f"帧 {frames_used}/{max_frames}）")
    else:
        msg = (f"⚠ token 估算 {total} 已超过上限 {max_tokens}（{ratio:.0%}），"
               f"继续发送会浪费额度甚至被模型拒绝。建议降级或缩小范围。")

    return {
        "over_budget": bool(over),
        "ratio": round(ratio, 4),
        "message": msg,
        "suggestions": suggestions,
    }
