"""OpenAI-compatible synchronous multipart STT, using only the Python standard library."""
from __future__ import annotations

import io
import json
import math
from urllib import error, request
import uuid
import wave

from stt_config import STTConfig


class STTAPIError(Exception):
    """Only sanitized messages should be raised by the HTTP layer."""


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not redirect audio or credentials to a different endpoint.


def _post_audio(config: STTConfig, audio: bytes, language=None):
    boundary = "vcb_" + uuid.uuid4().hex
    fields = {"model": config.api["model"], "response_format": config.api["response_format"]}
    if language:
        fields["language"] = language
    parts = []
    for key, value in fields.items():
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                      f"{value}\r\n").encode("utf-8"))
    parts.extend([
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
         "Content-Type: audio/wav\r\n\r\n").encode("ascii"),
        audio, f"\r\n--{boundary}--\r\n".encode("ascii"),
    ])
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json, text/plain"}
    key = config.api_key()
    if key:
        headers["Authorization"] = "Bearer " + key
    body = b"".join(parts)
    if len(body) > config.api["max_upload_mb"] * 1024 * 1024:
        raise STTAPIError("STT API multipart 请求超过 max_upload_mb；缩短模型名或提高上传限额")
    req = request.Request(config.endpoint(), data=body, headers=headers, method="POST")
    try:
        with request.build_opener(_NoRedirect()).open(req, timeout=config.api["timeout"]) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
    except error.HTTPError as exc:
        hints = {400: "检查模型与 response_format 是否兼容", 401: "检查密钥", 403: "检查权限",
                 404: "检查 base_url 和模型名", 413: "降低 max_upload_mb 或 chunk_seconds",
                 429: "限流或额度不足，请稍后手动重试"}
        raise STTAPIError(f"STT API HTTP {exc.code}：{hints.get(exc.code, '服务请求失败')}；未自动重试") from None
    except Exception:
        # Provider bodies, request headers, URLs, and exception reprs may contain secrets.
        raise STTAPIError("STT API 网络/超时/TLS 请求失败；请检查网络、证书和 timeout，未自动重试") from None
    if len(raw) > 8 * 1024 * 1024:
        raise STTAPIError("STT API 响应超过 8 MiB 限制")
    try:
        text = raw.decode("utf-8-sig")
        if config.api["response_format"] == "text":
            return {"text": text.strip()}
        data = json.loads(text)
    except (UnicodeError, ValueError):
        raise STTAPIError("STT API 返回非预期 JSON/UTF-8 响应；检查 response_format") from None
    if not isinstance(data, dict) or "error" in data:
        raise STTAPIError("STT API 返回错误或非对象 JSON；检查服务配置")
    if not any(k in data for k in ("text", "segments", "words")):
        raise STTAPIError("STT API JSON 缺少 text/segments/words，不是兼容转写响应")
    return data


def normalize_response(data: dict, offset: float, duration: float):
    """Preserve service times; text-only results are explicitly coarse chunk windows."""
    values, source = data.get("segments"), "api_segment"
    if not values and data.get("words"):
        values, source = data["words"], "api_word"
    if values is not None and not isinstance(values, list):
        raise STTAPIError("STT API segments/words 必须为列表")
    segments = []
    for item in values or []:
        if not isinstance(item, dict):
            raise STTAPIError("STT API 时间戳片段格式无效")
        text = item.get("text", item.get("word", ""))
        if not isinstance(text, str):
            raise STTAPIError("STT API 片段 text 必须为字符串")
        if not text.strip():
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            raise STTAPIError("STT API 片段缺少有效 start/end，无法精确对齐") from None
        if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end <= start or end > duration + 0.5:
            raise STTAPIError("STT API 时间戳超出音频块范围或顺序无效")
        end = min(end, duration)
        if start >= end:
            raise STTAPIError("STT API 时间戳超出音频块范围")
        segment = {"start": round(offset + start, 3), "end": round(offset + end, 3),
                   "text": text.strip(), "timestamp_source": source}
        if isinstance(item.get("speaker"), str):
            segment["speaker"] = item["speaker"]
        segments.append(segment)
    if not segments:
        text = data.get("text", "")
        if not isinstance(text, str):
            raise STTAPIError("STT API text 必须为字符串")
        if text.strip():
            segments = [{"start": round(offset, 3), "end": round(offset + duration, 3),
                         "text": text.strip(), "timestamp_source": "chunk"}]
    return sorted(segments, key=lambda s: (s["start"], s["end"]))


def transcribe_wav_api(wav_path: str, config: STTConfig, language=None, offset: float = 0.0) -> dict:
    """Bound each upload by time and bytes. No retries or provider/model fallback."""
    result = {"status": "ok", "reason": "", "backend": "api", "model": config.api["model"],
              "language": language, "segments": [], "text": "", "char_count": 0, "truncated": False,
              "timestamp_precision": "segment", "warnings": [],
              "api_usage": {"requests": 0, "completed_chunks": 0, "attempted_audio_seconds": 0.0,
                            "completed_audio_seconds": 0.0,
                            "reported_usage": [], "note": "STT 服务用量独立于看图模型 token 预算；失败请求也可能计费。"}}
    try:
        config.validate_api()
        with wave.open(str(wav_path), "rb") as src:
            params = src.getparams()
            bytes_per_frame = src.getnchannels() * src.getsampwidth()
            # Leave space for WAV and multipart headers inside the configured upload cap.
            byte_frames = int((config.api["max_upload_mb"] * 1024 * 1024 - 4096) // bytes_per_frame)
            chunk_frames = min(int(config.api["chunk_seconds"] * src.getframerate()), byte_frames)
            if chunk_frames <= 0:
                raise STTAPIError("API 上传/时长限制过小，无法容纳音频块")
            result["api_usage"]["total_chunks"] = math.ceil(src.getnframes() / chunk_frames)
            position = 0
            while position < src.getnframes():
                count = min(chunk_frames, src.getnframes() - position)
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as dest:
                    dest.setparams(params)
                    dest.writeframes(src.readframes(count))
                seconds = count / src.getframerate()
                result["api_usage"]["requests"] += 1
                result["api_usage"]["attempted_audio_seconds"] += seconds
                data = _post_audio(config, buffer.getvalue(), language)
                segments = normalize_response(data, offset + position / src.getframerate(), seconds)
                result["segments"].extend(segments)
                result["api_usage"]["completed_chunks"] += 1
                result["api_usage"]["completed_audio_seconds"] += seconds
                if isinstance(data.get("language"), str) and not result["language"]:
                    result["language"] = data["language"]
                usage = data.get("usage")
                if isinstance(usage, dict):
                    result["api_usage"]["reported_usage"].append({
                        k: v for k, v in usage.items() if k in ("seconds", "input_tokens", "output_tokens", "total_tokens")
                        and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    })
                position += count
    except STTAPIError as exc:
        result.update(status="error", reason=str(exc))
    except Exception:
        result.update(status="error", reason="STT API 音频读取或配置失败；检查 WAV 与服务配置")
    if any(s["timestamp_source"] == "chunk" for s in result["segments"]):
        result["timestamp_precision"] = "chunk"
        result["warnings"].append("API 未返回细粒度时间戳，只有音频块范围；不能用作精确台词时刻或动作对齐。")
    if result["status"] == "error" and result["segments"]:
        result["partial"] = True
        result["warnings"].append("API 转写中途失败，仅提供已成功音频块的内容；其余音频未转写。")
    result["text"] = " ".join(s["text"] for s in result["segments"])
    result["char_count"] = len(result["text"])
    for field in ("attempted_audio_seconds", "completed_audio_seconds"):
        result["api_usage"][field] = round(result["api_usage"][field], 3)
    return result
