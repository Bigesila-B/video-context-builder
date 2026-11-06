"""Budget/coverage regressions; real media tests use synthetic video and no model downloads."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_sampling import (
    choose_details, detail_candidates, duration_policy, evenly_spaced,
    plan_adaptive_sampling, select_transcript, token_allocation,
)
from build_video_context import build_video_context
from ffmpeg_utils import ffmpeg
from token_estimator import estimate_pack


class PlanningTests(unittest.TestCase):
    def test_duration_and_budget_matrix(self):
        for duration in (3, 30, 60, 300, 1200, 3600, 7200):
            for tokens in (4000, 8000, 16000, 50000):
                for width, height in ((1920, 1080), (1080, 1920)):
                    with self.subTest(duration=duration, tokens=tokens, size=(width, height)):
                        meta = dict(duration=duration, fps=30, width=width, height=height)
                        a = token_allocation(tokens)
                        p = plan_adaptive_sampling(meta, tokens, a, "test prompt")
                        self.assertLessEqual(p["estimate"]["total"], a["input_limit"])
                        self.assertLessEqual(p["frames_total"], p["max_frames"])
                        self.assertGreater(p["overview_frame_count"], 0)
                        self.assertGreaterEqual(p["overview_interval"], 1 / 30)
                        self.assertTrue(all(0 <= t < duration for t in p["second_seconds"]))
                        self.assertEqual(sum(a[k] for k in ("input_limit", "response_reserved",
                                         "safety_reserved", "revisit_reserved")), tokens)

    def test_auto_budget_grows_sublinearly(self):
        short, long = duration_policy(60), duration_policy(3600)
        self.assertLess(short["target_interval"], long["target_interval"])
        self.assertLess(short["suggested_max_tokens"], long["suggested_max_tokens"])
        self.assertLess(long["suggested_max_tokens"], short["suggested_max_tokens"] * 60)
        self.assertLessEqual(duration_policy(86400)["suggested_max_tokens"], 40000)

    def test_explicit_small_frame_limits(self):
        for cap in (1, 5, 6, 7, 20):
            p = plan_adaptive_sampling(dict(duration=120, fps=25, width=640, height=360),
                                      8000, token_allocation(8000), "question", max_frames=cap)
            self.assertLessEqual(p["frames_total"], cap)

    def test_budget_reduction_preserves_tail(self):
        self.assertEqual(evenly_spaced(list(range(100)), 4), [0, 33, 66, 99])
        meta = dict(duration=3600, fps=25, width=640, height=360)
        p = plan_adaptive_sampling(meta, 4000, token_allocation(4000), "question")
        n, step = p["overview_frame_count"], p["overview_interval"]
        self.assertGreater(n, 1)
        self.assertLess(3600 - (n - 1) * step, step + 1)

    def test_question_changes_detail_selection(self):
        segs = [{"start": 109, "end": 111, "text": "这里拿起钥匙"}]
        with_query = detail_candidates(0, 120, [], segs, "什么时候拿起钥匙", 2)
        selected = choose_details(with_query, 3, 0, 120, 2)
        self.assertTrue(any(c["source"] == "question_keyword" for c in selected))
        self.assertTrue(any(c["start"] < 40 for c in selected))

    def test_transcript_keeps_verbatim_timestamps_and_late_relevance(self):
        segs = [dict(start=i * 3, end=i * 3 + 2, text=f"普通对白第{i}句") for i in range(100)]
        segs[-2]["text"] = "钥匙放进抽屉了"
        selected = select_transcript(segs, 300, "钥匙在哪里", max_chars=500)
        self.assertIn(segs[-2], selected)
        self.assertTrue(all(s in segs for s in selected))
        self.assertTrue(any(s["start"] < 30 for s in selected))
        self.assertLessEqual(sum(len(s["text"]) for s in selected), 500)
        self.assertEqual(selected, sorted(selected, key=lambda s: s["start"]))

    def test_custom_range_uses_range_duration(self):
        meta = dict(duration=7200, fps=25, width=640, height=360)
        p = plan_adaptive_sampling(meta, 8000, token_allocation(8000), "question", custom_range=(120, 180))
        self.assertEqual(p["duration_policy"]["target_interval"], 0.5)
        self.assertTrue(all(120 <= t <= 178 for t in p["second_seconds"]))

    def test_selected_transcript_fits_prompt_segment_capacity(self):
        segs = [dict(start=i, end=i + 0.8, text="好") for i in range(900)]
        selected = select_transcript(segs, 100000, None, max_chars=10000)
        self.assertLessEqual(len(selected), 400)
        self.assertEqual(selected[-1], segs[-1])

    def test_invalid_reserves_and_duration(self):
        for tokens, kwargs in ((0, {}), (200, {}), (8000, {"safety_ratio": float("nan")}),
                               (8000, {"revisit_ratio": 1}), (8000, {"response_tokens": -1})):
            with self.subTest(tokens=tokens, kwargs=kwargs), self.assertRaises(ValueError):
                token_allocation(tokens, **kwargs)
        for span in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                duration_policy(span)


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="vc_adaptive_test_")
        cls.work = Path(cls.temp.name)
        cls.video = cls.work / "sample.mp4"
        p = ffmpeg("-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                   "testsrc2=size=320x180:rate=12:duration=12", "-c:v", "libx264", str(cls.video))
        if p.returncode:
            raise RuntimeError(p.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, name, **kwargs):
        return build_video_context(str(self.video), out=str(self.work / f"{name}.json"),
                                   progress=False, **kwargs)

    def check_payload(self, ctx):
        count = sum(s["cells"] for s in ctx["overview_sheets"]) + sum(s["frames"] for s in ctx["second_sheets"])
        count += len(ctx["original_frames"])
        self.assertEqual(ctx["sampling_plan"]["frames_total"], count)
        estimated = estimate_pack(ctx["overview_sheets"], ctx["second_sheets"], ctx["prompt"], [],
                                  ctx["original_frames"], model=ctx["model_preset"])
        self.assertEqual(ctx["token_estimate"]["total"], estimated["total"])
        for sheet in ctx["overview_sheets"] + ctx["second_sheets"]:
            self.assertTrue(Path(sheet["path"]).is_file())
        self.assertEqual(ctx["budget"]["over_budget"], estimated["total"] > ctx["budget"]["input_limit"])

    def test_default_duration_budget_and_real_assets(self):
        ctx = self.build("auto")
        self.check_payload(ctx)
        self.assertEqual(ctx["sampling_plan"]["strategy"], "adaptive")
        self.assertLess(ctx["budget"]["max_tokens"], 50000)
        self.assertTrue(ctx["budget"]["ready_to_send"])
        self.assertLessEqual(ctx["coverage"]["largest_unsampled_gap_sec"], 2)
        self.assertEqual(ctx["token_breakdown"]["transcript"], 0)

    def test_tight_budget_and_no_degrade(self):
        compact = self.build("tight", max_tokens=2500)
        self.check_payload(compact)
        self.assertLessEqual(compact["token_estimate"]["total"], compact["budget"]["input_limit"])
        full = self.build("full", max_tokens=2500, allow_degrade=False)
        self.check_payload(full)
        self.assertTrue(full["budget"]["over_budget"])
        self.assertFalse(full["budget"]["ready_to_send"])

    def test_range_boundary_and_alternate_layout(self):
        ctx = self.build("range", custom_range=(2, 5.2), max_tokens=8000,
                         second_layout="2x2", second_frames=4)
        self.check_payload(ctx)
        for s in ctx["second_sheets"]:
            self.assertEqual(s["layout"], "2x2")
            self.assertLessEqual(s["end"], 5.2)
            self.assertTrue(all(2 <= t < 5.2 for t in s["frame_timestamps"]))

    def test_full_transcript_preserved_selection_in_range(self):
        segs = [dict(start=i, end=i + 0.8, text=f"原始对白第{i}句") for i in range(12)]
        # Probe real audio-bearing metadata to exercise the STT orchestration without a model download.
        from probe_video import probe_video
        meta = probe_video(str(self.video))
        meta["has_audio"] = True
        with patch("build_video_context.probe_video", return_value=meta), patch(
            "build_video_context.transcribe", return_value={"status": "ok", "reason": "",
                "segments": segs, "language": "zh", "truncated": False, "char_count": 100}):
            ctx = self.build("transcript", custom_range=(4, 8), stt=True, max_tokens=8000)
        self.check_payload(ctx)
        self.assertTrue(all(s["start"] < 8 and s["end"] > 4 for s in ctx["transcript"]))
        source = json.loads(Path(ctx["transcript_info"]["full_transcript_path"]).read_text(encoding="utf-8"))
        self.assertEqual(source, segs)

    def test_too_large_fixed_question_blocks_send(self):
        ctx = self.build("question", max_tokens=2500, question="这是非常长的问题" * 3000)
        self.assertTrue(ctx["budget"]["over_budget"])
        self.assertFalse(ctx["budget"]["ready_to_send"])

    def test_invalid_inputs_fail_before_assets(self):
        for kwargs in (dict(custom_range=(20, 30)), dict(overview_interval=-1),
                       dict(second_frames=7), dict(max_frames=0), dict(max_tokens=-1),
                       dict(custom_range=(math.nan, 10)), dict(max_frames=2.5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build("invalid", **kwargs)

    def test_legacy_frame_limit_and_range_coverage(self):
        quick = self.build("quick_one", mode="quick", max_frames=1)
        self.assertLessEqual(quick["sampling_plan"]["frames_total"], 1)
        ranged = self.build("fixed_range", mode="custom-range", custom_range=(0, 12), max_frames=20)
        self.check_payload(ranged)
        self.assertLessEqual(ranged["sampling_plan"]["frames_total"], 20)
        self.assertTrue(any(s["start"] >= 6 for s in ranged["second_sheets"]))


if __name__ == "__main__":
    unittest.main()
