"""
STT 音频转写模块（可选）。

- 有音轨时用 faster-whisper 转写，输出 [{start, end, text}, ...]；
- 无音轨自动跳过（返回空列表 + 原因）；
- 未安装 faster-whisper 时优雅降级（不报错，返回 skipped 状态）；
- 长转写支持压缩（截断 + 分段摘要提示），避免文本 token 爆炸。

用法：
    python scripts/transcribe.py input.mp4 --model small --json out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 允许脚本直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffmpeg_utils import FFmpegNotFoundError, ffmpeg  # noqa: E402


def extract_audio(video_path: str, wav_path: str, sample_rate: int = 16000) -> str:
    """用 ffmpeg 抽出 16kHz 单声道 WAV（whisper 的标准输入）。

    只落盘一个小音频文件，避免把整段视频读进内存。
    """
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        wav_path,
    )
    return wav_path


def _load_whisper():
    """延迟导入 faster-whisper，返回 (WhisperModel 类, 错误信息)。"""
    try:
        from faster_whisper import WhisperModel  # type: ignore

        return WhisperModel, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


_HUB_HINT = (
    "模型未能从 HuggingFace 拉取。常见原因与解法：\n"
    "  1) 网络无法直连 HF：设置国内镜像后重试 —— 环境变量 HF_ENDPOINT=https://hf-mirror.com\n"
    "  2) 本机代理做了 TLS 中间人（SSL: CERTIFICATE_VERIFY_FAILED）：\n"
    "     把代理根证书配上 —— SSL_CERT_FILE / REQUESTS_CA_BUNDLE 指向该 CA 的 .pem；\n"
    "     或先绕开代理下载模型。\n"
    "  3) 更稳的做法：预先把模型下载到本地目录，然后 --whisper-model 直接传该目录路径，\n"
    "     例如 --whisper-model D:/models/faster-whisper-small（无需联网）。"
)


def _friendly_error(exc: Exception) -> str:
    """把底层异常翻译成「可操作」的提示。"""
    msg = str(exc)
    low = msg.lower()
    if ("certificate_verify_failed" in low or "ssl" in low
            or "localentrynotfound" in low or "locate the files on the hub" in low
            or "connection" in low or "timeout" in low):
        return f"STT 执行失败：{msg}\n{_HUB_HINT}"
    return f"STT 执行失败：{msg}"


def transcribe(
    video_path: str,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "int8",
    language: Optional[str] = None,
    work_dir: Optional[str] = None,
    has_audio: bool = True,
    max_chars: Optional[int] = None,
) -> Dict:
    """对视频音轨做 STT。

    Args:
        model_size: tiny/base/small/medium/large-v3，越大越准越慢
        device: 'auto' / 'cpu' / 'cuda'
        compute_type: 'int8'（CPU 友好）/ 'float16'（GPU）/ 'int8_float16'
        language: 指定语言（如 'zh'）可提速并提升准确率；None 自动检测
        has_audio: 调用方探测到的音轨存在性；False 直接跳过
        max_chars: 可选，超过则压缩转写文本

    Returns:
        {
          "status": "ok" | "skipped" | "unavailable" | "error",
          "reason": "...",
          "language": "zh",
          "segments": [{"start":..,"end":..,"text":".."}, ...],
          "text": "完整文本",
          "char_count": N,
          "truncated": bool,
        }
    """
    if not has_audio:
        return {
            "status": "skipped",
            "reason": "视频无音轨，已自动跳过 STT",
            "language": None, "segments": [], "text": "",
            "char_count": 0, "truncated": False,
        }

    WhisperModel, err = _load_whisper()
    if WhisperModel is None:
        return {
            "status": "unavailable",
            "reason": f"未安装 faster-whisper，已跳过 STT（pip install faster-whisper）。原始错误：{err}",
            "language": None, "segments": [], "text": "",
            "char_count": 0, "truncated": False,
        }

    work = Path(work_dir or Path(video_path).parent / "_vc_tmp_audio")
    wav = work / "audio16k.wav"
    try:
        extract_audio(video_path, str(wav))
    except FFmpegNotFoundError as exc:
        return {"status": "error", "reason": f"抽音频失败：{exc}", "language": None,
                "segments": [], "text": "", "char_count": 0, "truncated": False}

    info = None
    segments: List[Dict] = []
    try:
        # device='auto' 时交给 ctranslate2 自己判断
        dev = "auto" if device == "auto" else device
        model = WhisperModel(model_size, device=dev, compute_type=compute_type)
        seg_iter, info = model.transcribe(
            str(wav),
            language=language,
            vad_filter=True,            # 跳过静音，减少幻觉
            beam_size=5,
            word_timestamps=False,
        )
        for s in seg_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            segments.append({
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "text": text,
            })
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": _friendly_error(exc), "language": None,
                "segments": [], "text": "", "char_count": 0, "truncated": False}
    finally:
        try:
            wav.unlink()
        except OSError:
            pass

    full_text = " ".join(s["text"] for s in segments).strip()
    truncated = False
    if max_chars and len(full_text) > max_chars:
        segments = compress_segments(segments, max_chars=max_chars)
        full_text = " ".join(s["text"] for s in segments).strip()
        truncated = True

    return {
        "status": "ok",
        "reason": "",
        "language": getattr(info, "language", None) if info is not None else None,
        "segments": segments,
        "text": full_text,
        "char_count": len(full_text),
        "truncated": truncated,
    }


def compress_segments(segments: List[Dict], max_chars: int = 4000) -> List[Dict]:
    """长转写压缩：按「每段按比例抽稀」保留，同时保证时间戳仍覆盖全程。

    策略：把连续段合并成更粗的时间桶（默认 15 秒一桶），每桶只留一句最有代表性的文本
    （取桶内最长的句子），从而在保留时间轴的同时把字数压到上限以内。

    注意：这只是「压缩」，真正的语义摘要应由目标模型在 map-reduce 阶段完成。
    """
    if not segments:
        return []

    total = sum(len(s["text"]) for s in segments)
    if total <= max_chars:
        return segments

    bucket_sec = max(5.0, (segments[-1]["end"] - segments[0]["start"]) / max(1, (len(segments) // 8 or 1)))
    buckets: List[List[Dict]] = []
    cur: List[Dict] = []
    bucket_start = segments[0]["start"]
    for seg in segments:
        if seg["start"] - bucket_start >= bucket_sec and cur:
            buckets.append(cur)
            cur = []
            bucket_start = seg["start"]
        cur.append(seg)
    if cur:
        buckets.append(cur)

    picked: List[Dict] = []
    for b in buckets:
        best = max(b, key=lambda s: len(s["text"]))
        picked.append({
            "start": b[0]["start"],
            "end": b[-1]["end"],
            "text": best["text"] if len(b) == 1 else f"{best['text']}（本段共 {len(b)} 句，已压缩）",
        })

    # 仍然超限则整体再截断
    out: List[Dict] = []
    acc = 0
    for s in picked:
        if acc + len(s["text"]) > max_chars:
            break
        out.append(s)
        acc += len(s["text"])
    return out or picked[:1]


def format_transcript_with_timestamps(segments: List[Dict], max_lines: int = 400) -> str:
    """把 segments 渲染成带时间戳的文本块，供提示词使用。"""
    from make_contact_sheet import format_timestamp  # 局部导入，避免循环依赖

    lines = []
    for s in segments[:max_lines]:
        lines.append(f"[{format_timestamp(s['start'])}-{format_timestamp(s['end'])}] {s['text']}")
    if len(segments) > max_lines:
        lines.append(f"...（其余 {len(segments) - max_lines} 段已省略）")
    return "\n".join(lines)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="视频音频转写（faster-whisper，可选）")
    ap.add_argument("video")
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--json", default=None, help="把结果写到该 JSON 文件")
    args = ap.parse_args()

    res = transcribe(
        args.video, model_size=args.model, device=args.device,
        compute_type=args.compute_type, language=args.language,
        max_chars=args.max_chars,
    )
    print(f"状态: {res['status']}  {res['reason']}")
    print(f"段数: {len(res['segments'])}  字数: {res['char_count']}")
    for s in res["segments"][:20]:
        print(f"  [{s['start']:.2f}-{s['end']:.2f}] {s['text']}")

    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
