"""Real multipart HTTP tests against loopback only. No provider calls or model downloads."""
from __future__ import annotations

from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_sampling import detail_candidates
from build_video_context import build_video_context
from ffmpeg_utils import ffmpeg
from prompt_builder import build_transcript_block
from stt_api import normalize_response, STTAPIError, transcribe_wav_api
from stt_config import load_stt_config
from transcribe import transcribe


class Handler(BaseHTTPRequestHandler):
    requests_seen = []
    replies = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        msg = BytesParser(policy=policy.default).parsebytes(
            ("Content-Type: " + self.headers["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + raw)
        fields, audio = {}, b""
        for part in msg.iter_parts():
            name = part.get_param("name", header="Content-Disposition")
            if name == "file":
                audio = part.get_payload(decode=True)
            else:
                fields[name] = part.get_payload(decode=True).decode()
        with wave.open(io.BytesIO(audio), "rb") as src:
            duration = src.getnframes() / src.getframerate()
        Handler.requests_seen.append({"path": self.path, "fields": fields, "duration": duration,
                                      "size": len(raw), "authorization": self.headers.get("Authorization")})
        reply = Handler.replies.pop(0) if Handler.replies else {}
        status = reply.get("status", 200)
        body = reply.get("body", {"segments": [{"start": 0, "end": min(duration, 0.25), "text": "测试对白"}],
                                  "language": "zh", "usage": {"seconds": duration}})
        payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        for name, value in reply.get("headers", {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class STTAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work_context = tempfile.TemporaryDirectory(prefix="vc_stt_api_test_")
        cls.work = Path(cls.work_context.name)
        cls.wav = cls.work / "audio.wav"
        with wave.open(str(cls.wav), "wb") as out:
            out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            out.writeframes(b"\x00\x00" * 48000)
        cls.video = cls.work / "audio_video.mp4"
        proc = ffmpeg("-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                      "testsrc2=size=160x120:rate=12:duration=3", "-i", str(cls.wav),
                      "-c:v", "libx264", "-c:a", "aac", "-shortest", str(cls.video))
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.work_context.cleanup()

    def setUp(self):
        self.env = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("STT_")}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        Handler.requests_seen, Handler.replies = [], []

    def config(self, **kwargs):
        return load_stt_config(backend="api", api_options={"base_url": self.base, "model": "mock-asr", **kwargs})

    def test_multipart_auth_fields_and_usage(self):
        os.environ["STT_TEST_SECRET"] = "fake-test-key-not-a-real-credential"
        c = self.config(api_key_env="STT_TEST_SECRET")
        result = transcribe_wav_api(str(self.wav), c, language="zh")
        self.assertEqual(result["status"], "ok")
        seen = Handler.requests_seen[0]
        self.assertEqual(seen["path"], "/v1/audio/transcriptions")
        self.assertEqual(seen["authorization"], "Bearer fake-test-key-not-a-real-credential")
        self.assertEqual(seen["fields"], {"model": "mock-asr", "response_format": "verbose_json", "language": "zh"})
        self.assertEqual(result["api_usage"]["completed_chunks"], 1)
        self.assertEqual(result["api_usage"]["reported_usage"], [{"seconds": 3.0}])
        self.assertNotIn(os.environ["STT_TEST_SECRET"], json.dumps(result))
        self.assertNotIn(os.environ["STT_TEST_SECRET"], json.dumps(c.public_summary()))

    def test_chunk_offsets_and_upload_cap(self):
        result = transcribe_wav_api(str(self.wav), self.config(chunk_seconds=1), offset=120)
        self.assertEqual([s["start"] for s in result["segments"]], [120, 121, 122])
        self.assertEqual(result["api_usage"]["requests"], 3)
        Handler.requests_seen = []
        result = transcribe_wav_api(str(self.wav), self.config(max_upload_mb=0.02))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(Handler.requests_seen), 1)
        self.assertTrue(all(r["size"] <= 0.02 * 1024 * 1024 for r in Handler.requests_seen))
        self.assertAlmostEqual(sum(r["duration"] for r in Handler.requests_seen), 3)

    def test_text_only_response_is_coarse_and_not_used_as_speech_onset(self):
        Handler.replies = [{"body": {"text": "只返回文字"}}]
        result = transcribe_wav_api(str(self.wav), self.config(response_format="json"), offset=60)
        self.assertEqual(result["timestamp_precision"], "chunk")
        self.assertEqual(result["segments"][0]["start"], 60)
        self.assertEqual(result["segments"][0]["end"], 63)
        self.assertEqual(result["segments"][0]["timestamp_source"], "chunk")
        self.assertIn("非逐句时间", build_transcript_block(result["segments"]))
        candidates = detail_candidates(60, 70, [], result["segments"], "文字", 2)
        self.assertTrue(all(c["source"] == "uniform" for c in candidates))

    def test_plain_text_format(self):
        Handler.replies = [{"body": "纯文本返回".encode("utf-8")}]
        result = transcribe_wav_api(str(self.wav), self.config(response_format="text"))
        self.assertEqual(result["text"], "纯文本返回")
        self.assertEqual(result["timestamp_precision"], "chunk")

    def test_words_and_speaker_normalization(self):
        words = normalize_response({"words": [{"start": 0.2, "end": 0.4, "word": "hello"}]}, 10, 1)
        self.assertEqual(words[0]["start"], 10.2)
        self.assertEqual(words[0]["timestamp_source"], "api_word")
        seg = normalize_response({"segments": [{"start": 0, "end": 0.5, "text": "你好", "speaker": "A"}]}, 0, 1)
        self.assertEqual(seg[0]["speaker"], "A")

    def test_invalid_timestamps_rejected(self):
        for start, end in ((-1, 1), (2, 1), (0, 100), (math.nan, 1), (0, math.inf)):
            with self.subTest(start=start, end=end), self.assertRaises(STTAPIError):
                normalize_response({"segments": [{"start": start, "end": end, "text": "x"}]}, 0, 3)

    def test_http_errors_no_retry_or_secret_echo(self):
        for code in (400, 401, 403, 404, 413, 429, 500):
            with self.subTest(code=code):
                Handler.replies = [{"status": code, "body": {"error": "SENSITIVE_PROVIDER_BODY"}}]
                count = len(Handler.requests_seen)
                result = transcribe_wav_api(str(self.wav), self.config())
                self.assertEqual(result["status"], "error")
                self.assertEqual(len(Handler.requests_seen), count + 1)
                self.assertNotIn("SENSITIVE_PROVIDER_BODY", json.dumps(result))
                self.assertIn(str(code), result["reason"])

    def test_redirect_does_not_resend_audio(self):
        Handler.replies = [{"status": 307, "headers": {"Location": self.base + "/other"}, "body": b""}]
        result = transcribe_wav_api(str(self.wav), self.config())
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(Handler.requests_seen), 1)

    def test_partial_failure_preserves_completed_evidence(self):
        Handler.replies = [{}, {"status": 500}]
        result = transcribe_wav_api(str(self.wav), self.config(chunk_seconds=1))
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["partial"])
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["api_usage"]["requests"], 2)
        self.assertEqual(result["api_usage"]["completed_chunks"], 1)

    def test_invalid_json_fails_without_echo(self):
        Handler.replies = [{"body": b"not-json SENSITIVE_PROVIDER_BODY"}]
        result = transcribe_wav_api(str(self.wav), self.config())
        self.assertEqual(result["status"], "error")
        self.assertNotIn("SENSITIVE_PROVIDER_BODY", json.dumps(result))

    def test_timeout_is_sanitized_and_not_retried(self):
        with patch("stt_api.request.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError("secret-body-do-not-log")
            result = transcribe_wav_api(str(self.wav), self.config())
        self.assertEqual(result["status"], "error")
        self.assertEqual(opener.return_value.open.call_count, 1)
        self.assertNotIn("secret-body-do-not-log", json.dumps(result))

    def test_missing_response_fields_and_empty_speech(self):
        Handler.replies = [{"body": {"unrelated": "value"}}]
        result = transcribe_wav_api(str(self.wav), self.config())
        self.assertEqual(result["status"], "error")
        Handler.replies = [{"body": {"text": ""}}]
        result = transcribe_wav_api(str(self.wav), self.config())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["segments"], [])

    def test_api_char_limit_preserves_original_timestamps(self):
        payload = {"segments": [{"start": 0, "end": 0.4, "text": "first"},
                                 {"start": 1, "end": 1.4, "text": "middle"},
                                 {"start": 2, "end": 2.4, "text": "last"}]}
        Handler.replies = [{"body": payload}]
        result = transcribe(str(self.video), backend="api", max_chars=10,
                            api_options={"base_url": self.base, "model": "mock-asr"})
        self.assertTrue(result["truncated"])
        self.assertTrue(all(any(s["start"] == p["start"] and s["end"] == p["end"] and s["text"] == p["text"]
                                for p in payload["segments"]) for s in result["segments"]))
        self.assertTrue(all(s["timestamp_source"] == "api_segment" for s in result["segments"]))

    def test_missing_remote_credentials_fails_before_audio(self):
        with patch("transcribe.extract_audio") as extract:
            result = transcribe("unused.mp4", backend="api", api_options={"base_url": "https://example.invalid/v1", "model": "x"})
        self.assertEqual(result["status"], "error")
        self.assertIn("STT_API_KEY", result["reason"])
        extract.assert_not_called()

    def test_no_audio_never_loads_or_requests(self):
        with patch("transcribe._load_whisper") as load:
            result = transcribe("unused.mp4", has_audio=False, backend="api")
        self.assertEqual(result["status"], "skipped")
        self.assertFalse(Handler.requests_seen)
        load.assert_not_called()

    def test_config_precedence_and_secret_field_rejection(self):
        path = self.work / "config.json"
        path.write_text(json.dumps({"backend": "api", "api": {"base_url": self.base, "model": "from-file"}}), encoding="utf-8")
        os.environ["STT_API_MODEL"] = "from-env"
        self.assertEqual(load_stt_config(str(path)).api["model"], "from-env")
        self.assertEqual(load_stt_config(str(path), api_options={"model": "from-arg"}).api["model"], "from-arg")
        path.write_text(json.dumps({"api": {"api_key": "do-not-store-secrets"}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_stt_config(str(path))

    def test_custom_key_environment_variable(self):
        os.environ["STT_API_KEY_ENV"] = "STT_CUSTOM_SECRET"
        os.environ["STT_CUSTOM_SECRET"] = "fake-custom-key"
        result = transcribe_wav_api(str(self.wav), self.config())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(Handler.requests_seen[0]["authorization"], "Bearer fake-custom-key")
        self.assertNotIn("fake-custom-key", json.dumps(result))

    def test_invalid_config_validation(self):
        for options in ({"base_url": "https://user:pass@example.invalid"},
                        {"base_url": "https://example.invalid?key=secret"}, {"model": ""},
                        {"timeout": math.nan}, {"chunk_seconds": 0}, {"response_format": "srt"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.config(**options)

    def test_api_dispatch_no_local_dependency_and_time_range(self):
        work = self.work / "dispatch"
        with patch("transcribe._load_whisper", side_effect=AssertionError("must not load local model")):
            result = transcribe(str(self.video), backend="api", work_dir=str(work),
                                api_options={"base_url": self.base, "model": "mock-asr", "chunk_seconds": 1},
                                time_range=(1, 3))
        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["start"] for s in result["segments"]], [1, 2])
        self.assertAlmostEqual(result["api_usage"]["attempted_audio_seconds"], 2)
        self.assertEqual(list(work.iterdir()), [])

    def test_local_dispatch_uses_config_without_network(self):
        with patch("transcribe._transcribe_local", return_value={"status": "ok", "text": "", "segments": [],
                   "char_count": 0, "truncated": False}) as local:
            result = transcribe(str(self.video), backend="local", model_size="base", device="cpu")
        self.assertEqual(result["backend"], "local")
        self.assertEqual(local.call_args.kwargs["model_size"], "base")
        self.assertEqual(local.call_args.kwargs["device"], "cpu")
        self.assertFalse(Handler.requests_seen)

    def test_config_check_cli_has_no_network(self):
        path = self.work / "check.json"
        path.write_text(json.dumps({"backend": "api", "api": {"base_url": self.base, "model": "mock-asr"}}), encoding="utf-8")
        proc = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).with_name("transcribe.py")),
                               "--check-config", "--stt-config", str(path)], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["backend"], "api")
        self.assertFalse(Handler.requests_seen)

    def test_context_builder_integration(self):
        result = build_video_context(str(self.video), mode="auto", stt=True, custom_range=(1, 3),
                                     stt_backend="api", stt_api_options={"base_url": self.base, "model": "mock-asr"},
                                     out=str(self.work / "context.json"), progress=False)
        self.assertEqual(result["transcript_info"]["backend"], "api")
        self.assertEqual(result["transcript_info"]["status"], "ok")
        self.assertEqual(result["transcript"][0]["start"], 1)
        self.assertEqual(result["transcript_info"]["api_usage"]["completed_chunks"], 1)
        self.assertIn("测试对白", result["prompt"])
        self.assertTrue(result["budget"]["ready_to_send"])

    def test_coarse_context_explains_timestamp_limits(self):
        Handler.replies = [{"body": {"text": "这是一段没有精确时间的对白"}}]
        result = build_video_context(str(self.video), stt=True, stt_backend="api",
                                     stt_api_options={"base_url": self.base, "model": "mock-asr", "response_format": "json"},
                                     out=str(self.work / "coarse_context.json"), progress=False)
        self.assertEqual(result["transcript_info"]["timestamp_precision"], "chunk")
        self.assertIn("不能据此精确定位", result["prompt"])
        self.assertTrue(any("细粒度时间戳" in warning for warning in result["warnings"]))

    def test_builder_cli_uses_saved_api_config(self):
        config_path = self.work / "cli-stt.json"
        config_path.write_text(json.dumps({"backend": "api", "api": {
            "base_url": self.base, "model": "configured-asr", "api_key_env": "STT_CLI_SECRET"}}), encoding="utf-8")
        os.environ["STT_CLI_SECRET"] = "fake-cli-key"
        out = self.work / "cli-context.json"
        proc = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).with_name("build_video_context.py")),
                               "--video", str(self.video), "--stt", "--stt-config", str(config_path),
                               "--quiet", "--out", str(out)], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        context = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(context["transcript_info"]["status"], "ok")
        self.assertEqual(context["transcript_info"]["model"], "configured-asr")
        self.assertEqual(Handler.requests_seen[0]["authorization"], "Bearer fake-cli-key")
        self.assertNotIn("fake-cli-key", proc.stdout + proc.stderr + out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
