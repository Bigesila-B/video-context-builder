"""
端到端自检 —— 覆盖需求里的全部验收标准。

思路：用 ffmpeg 合成「可程序化验证」的测试视频，再跑完整流水线，最后**从拼图里读回像素**
来反推时间戳，从而在无人工目检的情况下证明「时间戳准确到秒」。

测试视频的编码方式（关键设计）：
  - 第 n 秒的整帧底色 = ( (n%16)*16, (n//16)*16, 128 )  → 颜色唯一标识「第几秒」；
  - 帧内左上角 48x48 的灰块亮度 = 15 + k*8（k = 该秒内的第几帧）→ 亮度标识「秒内第几帧」。

于是从生成的秒图里：
  - 采样背景区   → 复原「秒」→ 校验 sheet.start 是否准确；
  - 采样左上灰块 → 复原「秒内帧号」→ 校验 1/秒 的采样间隔是否准确。

运行：
    python scripts/selfcheck.py            # 全部用例
    python scripts/selfcheck.py --keep     # 保留产物便于目检
    python scripts/selfcheck.py -k quick   # 只跑名字含 quick 的用例
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ffmpeg_utils import ffmpeg, ffmpeg_bin, ffprobe_bin  # noqa: E402
from make_contact_sheet import LABEL_BAR, PAD, _cell_size, parse_layout  # noqa: E402
from build_video_context import build_video_context  # noqa: E402

TITLE_H = 34

# --------------------------------------------------------------------------- #
# 测试视频合成
# --------------------------------------------------------------------------- #
def _second_bg(n: int) -> Tuple[int, int, int]:
    """第 n 秒的底色，颜色唯一映射到 n（0..255 秒内无歧义）。"""
    return ((n % 16) * 16, (n // 16 % 16) * 16, 128)


def _k_level(k: int) -> int:
    """秒内第 k 帧的灰块亮度。"""
    return 15 + k * 8


def make_test_video(
    out_path: str,
    seconds: int = 30,
    fps: int = 30,
    size: Tuple[int, int] = (320, 240),
    with_audio: bool = False,
    with_patch: bool = True,
    work_dir: Optional[str] = None,
) -> str:
    """合成测试视频（可程序化反推时间）。"""
    tmp = Path(work_dir or tempfile.mkdtemp(prefix="vc_gen_"))
    tmp.mkdir(parents=True, exist_ok=True)
    w, h = size
    patch = 48

    for n in range(seconds):
        bg = _second_bg(n)
        for k in range(fps):
            im = Image.new("RGB", (w, h), bg)
            if with_patch:
                lv = _k_level(k)
                for yy in range(patch):
                    for xx in range(patch):
                        im.putpixel((xx, yy), (lv, lv, lv))
            im.save(tmp / f"f_{n * fps + k:06d}.png")

    cmd = ["-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(fps), "-i", str(tmp / "f_%06d.png")]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [str(out_path)]
    ffmpeg(*cmd)

    if not work_dir:
        shutil.rmtree(tmp, ignore_errors=True)
    return str(out_path)


# --------------------------------------------------------------------------- #
# 从拼图里读回信息
# --------------------------------------------------------------------------- #
def _sheet_geometry(cols: int, rows: int, cell: int, src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    cw, ch = _cell_size(src_w, src_h, cell)
    return cw, ch, cw + PAD, ch + LABEL_BAR + PAD


def _region_mean(im: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """取一块区域的平均色：缩到 1x1 用 BOX 采样 = 精确均值（且不触发 getdata 弃用告警）。"""
    return im.crop(box).resize((1, 1), Image.BOX).getpixel((0, 0))  # type: ignore[return-value]


def read_cell(sheet_path: str, cols: int, cell: int, src_w: int, src_h: int, idx: int,
              rows: int = 1) -> Tuple[Tuple[int, int, int], int]:
    """读取拼图里第 idx 格：返回 (背景平均色, 左上灰块平均亮度)。"""
    cw, ch, block_w, block_h = _sheet_geometry(cols, rows, cell, src_w, src_h)
    r, c = divmod(idx, cols)
    x = PAD + c * block_w
    y = TITLE_H + PAD + r * block_h

    with Image.open(sheet_path) as im:
        im = im.convert("RGB")
        bg_mean = _region_mean(im, (x + cw // 2, y + ch // 2, x + cw - 6, y + ch - 6))
        lv = _region_mean(im, (x + 8, y + 8, x + 34, y + 34))[0]
    return bg_mean, lv


def recover_second(bg: Tuple[int, int, int]) -> int:
    """从底色复原「第几秒」。"""
    r = round(bg[0] / 16) % 16
    g = round(bg[1] / 16) % 16
    return g * 16 + r


def recover_k(level: int) -> int:
    """从灰块亮度复原「秒内第几帧」。"""
    return max(0, round((level - 15) / 8))


# --------------------------------------------------------------------------- #
# 断言小工具
# --------------------------------------------------------------------------- #
class Checker:
    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[str] = []

    def check(self, cond: bool, msg: str) -> bool:
        (self.passed if cond else self.failed).append(msg)
        print(("  [PASS] " if cond else "  [FAIL] ") + msg)
        return cond

    def eq(self, got, want, msg: str) -> bool:
        return self.check(got == want, f"{msg}（got={got!r}, want={want!r}）")

    def le(self, got, want, msg: str) -> bool:
        return self.check(got <= want, f"{msg}（got={got!r}, 应 <= {want!r}）")

    def ge(self, got, want, msg: str) -> bool:
        return self.check(got >= want, f"{msg}（got={got!r}, 应 >= {want!r}）")

    def contains(self, hay: str, needle: str, msg: str) -> bool:
        return self.check(needle in (hay or ""), f"{msg}（应包含 {needle!r}）")


# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #
_SHORT_VIDEO_CACHE: dict = {}


def ensure_short_video(work: Path) -> str:
    """确保 30 秒无音轨测试视频存在（多个用例共用，允许单独挑一个用例跑）。"""
    key = str(work)
    p = Path(_SHORT_VIDEO_CACHE.get(key) or (work / "short_30s_noaudio.mp4"))
    if not p.exists():
        make_test_video(str(p), seconds=30, fps=30, size=(320, 240), with_audio=False)
    _SHORT_VIDEO_CACHE[key] = str(p)
    return str(p)


def case_short_video(ck: Checker, work: Path, keep: bool) -> Dict:
    """验收 1/3/4/6：30 秒视频，无音轨 → 总览图 + 秒图 + prompt；时间戳准确；STT 自动跳过；临时文件清理。"""
    print("\n=== 用例 1：30 秒视频（无音轨）/ standard ===")
    video = Path(ensure_short_video(work))

    tmp_before = {p.name for p in Path(tempfile.gettempdir()).glob("vc_b_*")}
    t0 = time.time()
    ctx = build_video_context(
        str(video), mode="standard", stt=True, max_tokens=50000,
        out=str(work / "short.json"), out_dir=str(work / "short_assets"),
        progress=True,
    )
    dt = time.time() - t0

    ck.eq(ctx["meta"]["duration"] > 29.5 and ctx["meta"]["duration"] < 30.6, True, "元数据时长约 30s")
    ck.eq(ctx["meta"]["has_audio"], False, "正确识别「无音轨」")

    # --- 材料齐备 ---
    ck.ge(len(ctx["overview_sheets"]), 1, "生成了总览图")
    ck.ge(len(ctx["second_sheets"]), 1, "生成了秒图")
    ck.contains(ctx["prompt"], "你是一个视频理解助手", "prompt 含角色设定")
    ck.contains(ctx["prompt"], "只基于提供的材料回答", "prompt 含「只基于材料」约束")
    ck.contains(ctx["prompt"], "抽样帧中未看到", "prompt 含「不确定要说明」约束")
    ck.contains(ctx["prompt"], "不要编造未出现的情节", "prompt 含「禁止编造」约束")
    ck.contains(ctx["prompt"], "[MM:SS]", "prompt 要求引用时间戳")

    # --- 无音轨 → STT 自动跳过 ---
    ck.eq(ctx["transcript_info"]["status"], "skipped", "无音轨时 STT 自动跳过")
    ck.eq(ctx["transcript"], [], "transcript 为空数组")

    # --- 布局与帧数 ---
    for s in ctx["second_sheets"]:
        if s["layout"] != "3x2" or s["frames"] != 6:
            ck.check(False, f"秒图布局/帧数异常：{s['layout']} / {s['frames']}")
            break
    else:
        ck.check(True, f"全部 {len(ctx['second_sheets'])} 张秒图均为 3x2 / 6 帧")

    for s in ctx["overview_sheets"]:
        if s["layout"] != "4x4":
            ck.check(False, f"总览布局异常：{s['layout']}")
            break
    else:
        ck.check(True, f"全部 {len(ctx['overview_sheets'])} 张总览图均为 4x4")

    # --- 秒图文件存在 ---
    missing = [s["path"] for s in ctx["second_sheets"] + ctx["overview_sheets"] if not Path(s["path"]).exists()]
    ck.eq(missing, [], "所有拼图文件都已落盘")

    # --- 精度反查：从像素复原时间戳 ---
    src_w, src_h = 320, 240
    cols, _rows = parse_layout("3x2")
    errors_sec: List[str] = []
    errors_k: List[str] = []
    for s in ctx["second_sheets"]:
        fts = s.get("frame_timestamps", [])
        for i in range(len(fts)):
            bg, lv = read_cell(s["path"], cols, 224, src_w, src_h, i, rows=2)
            rec_sec = recover_second(bg)
            rec_k = recover_k(lv)
            exp_sec = int(fts[i])
            exp_k = round((fts[i] - exp_sec) * 30) % 30
            if rec_sec != exp_sec:
                errors_sec.append(f"{s['timestamp']} 第{i}帧: 图上是第{rec_sec}s，声明是第{exp_sec}s")
            if abs(rec_k - exp_k) > 1:
                errors_k.append(f"{s['timestamp']} 第{i}帧: 图上秒内第{rec_k}帧，声明第{exp_k}帧")
    ck.eq(errors_sec[:3], [], f"秒图的秒级时间戳与画面一致（共查 {sum(len(s['frame_timestamps']) for s in ctx['second_sheets'])} 帧）")
    ck.eq(errors_k[:3], [], "秒图的秒内采样位置与声明的 frame_timestamps 一致")

    # 采样间隔应恰好是 源帧率/每张帧数（30fps / 6 帧 = 每 5 帧取 1 帧 -> 0.1667s）
    gaps = []
    for s in ctx["second_sheets"]:
        fts = s.get("frame_timestamps", [])
        if len(fts) >= 2:
            gaps.append(round((fts[1] - fts[0]) * 30))
    ck.eq(sorted(set(gaps)), [5], "秒图采样间隔为 5 帧（= 30fps / 6 帧），不存在 fps 回落的偏差")

    # --- token 预算 ---
    ck.le(ctx["token_estimate"]["total"], 50000, "token 估算在上限内")
    ck.eq(ctx["budget"]["over_budget"], False, "未超预算")

    # --- 临时目录清理 ---
    tmp_after = {p.name for p in Path(tempfile.gettempdir()).glob("vc_b_*")}
    ck.eq(sorted(tmp_after - tmp_before), [], "临时抽帧目录已清理干净")

    print(f"  （用时 {dt:.1f}s，总览 {len(ctx['overview_sheets'])} 张，秒图 {len(ctx['second_sheets'])} 张，"
          f"token≈{ctx['token_estimate']['total']}）")
    return ctx


def case_quick_and_range(ck: Checker, work: Path) -> None:
    """验收 3 + 模式：quick 不出秒图；custom-range 只处理指定区间。"""
    print("\n=== 用例 2：quick 模式 / custom-range 模式 ===")
    video = ensure_short_video(work)

    q = build_video_context(str(video), mode="quick", max_tokens=50000,
                            out=str(work / "quick.json"), out_dir=str(work / "quick_assets"),
                            progress=False)
    ck.eq(q["mode"], "quick", "mode 落为 quick")
    ck.eq(len(q["second_sheets"]), 0, "quick 不生成秒图")
    ck.le(q["sampling_plan"]["frames_total"], 32, "quick 帧数 <= 32")
    ck.eq(q["transcript_info"]["status"], "skipped", "quick 默认不开 STT")

    r = build_video_context(str(video), mode="custom-range", custom_range=(12, 15), stt=False,
                            out=str(work / "range.json"), out_dir=str(work / "range_assets"),
                            progress=False)
    ck.eq(r["mode"], "custom-range", "mode 落为 custom-range")
    ck.eq(r["custom_range"], [12.0, 15.0], "custom_range 正确解析")
    ok = all(11.99 <= s["start"] <= 15.0 for s in r["second_sheets"]) and len(r["second_sheets"]) >= 2
    ck.check(ok, f"秒图全部落在 12-15s 内（共 {len(r['second_sheets'])} 张）")


def case_budget_guard(ck: Checker, work: Path) -> None:
    """验收 5：超预算必须提醒，并默认自动降级。"""
    print("\n=== 用例 3：token 超限提醒与降级 ===")
    video = ensure_short_video(work)

    a = build_video_context(str(video), mode="standard", stt=False, max_tokens=1200,
                            out=str(work / "tight.json"), out_dir=str(work / "tight_assets"),
                            progress=False, allow_degrade=True)
    ck.ge(len(a["warnings"]), 1, "超限时产生提醒")
    ck.check(any("秒图数量" in w or "上限" in w for w in a["warnings"]),
             "提醒中明确说明了降级/超限情况")
    ck.eq(a["sampling_plan"]["degraded"], True, "已触发降级")

    b = build_video_context(str(video), mode="standard", stt=False, max_tokens=1200,
                            out=str(work / "tight2.json"), out_dir=str(work / "tight2_assets"),
                            progress=False, allow_degrade=False)
    ck.eq(b["budget"]["over_budget"], True, "--no-degrade 时保留超限并如实报告")
    ck.ge(len(b["second_sheets"]), len(a["second_sheets"]), "--no-degrade 保留更多秒图")
    ck.check(a["token_estimate"]["total"] < b["token_estimate"]["total"],
             f"降级确实降低了 token（{a['token_estimate']['total']} < {b['token_estimate']['total']}）")
    ck.check(any("quick" in w for w in a["warnings"]) or any("quick" in s for s in b["budget"].get("suggestions", [])),
             "提醒里给出了模式选择建议（如 quick / --range）")


def case_long_video(ck: Checker, work: Path) -> None:
    """验收 2：10 分钟视频默认不出 600 张秒图，走总览 + 关键段 + map-reduce。"""
    print("\n=== 用例 4：10 分钟长视频（long-video / auto）===")
    video = work / "long_600s_noaudio.mp4"
    if not Path(video).exists():
        make_test_video(str(video), seconds=600, fps=4, size=(160, 120),
                        with_audio=False, with_patch=False)

    t0 = time.time()
    ctx = build_video_context(str(video), mode="auto", stt=False, max_tokens=50000,
                              out=str(work / "long.json"), out_dir=str(work / "long_assets"),
                              progress=True)
    dt = time.time() - t0

    ck.eq(ctx["mode"], "long-video", "auto 对 >10 分钟视频选择 long-video")
    ck.eq(ctx["meta"]["duration"] > 599.0, True, "时长约 600s")
    ck.le(len(ctx["second_sheets"]), 30, "没有全量生成 600 张秒图")
    ck.le(ctx["sampling_plan"]["frames_total"], 128, "总帧数受模式上限约束")
    ck.le(ctx["token_estimate"]["total"], 50000, "token 估算可控")
    ck.contains(ctx["prompt"], "抽样帧之外的时间段没有被观测", "prompt 里交代了抽样率与未观测区间")

    mr = ctx.get("map_reduce")
    ck.check(bool(mr), "输出包含 map_reduce 分段计划")
    if mr:
        ck.eq(mr["chunk_count"], 5, "600s / 120s = 5 段")
        ck.ge(len(mr["chunks"]), 5, "chunks 已展开")
        ck.contains(mr["chunks"][0]["map_prompt"], "本段", "每段有 map 提示词")
        ck.contains(mr["reduce_prompt_template"], "分段摘要", "有 reduce 提示词模板")
        covered = all(11.99 <= c["start"] <= 600.0 for c in ctx["second_sheets"])
        ck.check(covered, "秒图起点都在视频时长内")

    print(f"  （用时 {dt:.1f}s，总览 {len(ctx['overview_sheets'])} 张，秒图 {len(ctx['second_sheets'])} 张，"
          f"token≈{ctx['token_estimate']['total']}）")


def case_stt_paths(ck: Checker, work: Path) -> None:
    """验收 4：有音轨时走 STT（装了 faster-whisper 就转写，没装则优雅降级）。"""
    print("\n=== 用例 5：有音轨视频的 STT 路径 ===")
    video = work / "short_12s_audio.mp4"
    make_test_video(str(video), seconds=12, fps=30, size=(320, 240),
                    with_audio=True, with_patch=False)

    ctx = build_video_context(str(video), mode="standard", stt=True, max_tokens=50000,
                              out=str(work / "audio.json"), out_dir=str(work / "audio_assets"),
                              whisper_model="tiny", progress=True)

    ck.eq(ctx["meta"]["has_audio"], True, "正确识别「有音轨」")
    status = ctx["transcript_info"]["status"]
    ck.check(status in ("ok", "unavailable", "error"),
             f"有音轨时 STT 走正常路径（status={status}）")
    if status == "unavailable":
        print("      注：本机未安装 faster-whisper，已按设计优雅降级（不报错）。")
    if status == "ok":
        ck.check(isinstance(ctx["transcript"], list), "transcript 为 segments 列表")
        if ctx["transcript"]:
            seg = ctx["transcript"][0]
            ck.check(all(k in seg for k in ("start", "end", "text")),
                     "segment 含 start/end/text 三个字段")


def case_stt_pipeline(ck: Checker, work: Path) -> None:
    """STT 后处理链路单测（不需要下载模型）：时间戳渲染 / 长转写压缩 / 进入提示词与 token 估算。"""
    print("\n=== 用例 6：STT 后处理链路（离线可跑，不依赖模型下载）===")
    from transcribe import compress_segments, format_transcript_with_timestamps
    from prompt_builder import build_map_prompt, build_prompt, build_reduce_prompt
    from token_estimator import estimate_text_tokens

    segs = [
        {"start": 0.0, "end": 2.34, "text": "大家好，今天讲一下视频理解。"},
        {"start": 2.34, "end": 5.1, "text": "首先我们要把视频转成模型能看懂的图片。"},
        {"start": 12.0, "end": 15.0, "text": "注意这里的秒图对应视频中的一秒。"},
        {"start": 3725.0, "end": 3728.0, "text": "最后总结一下整体流程。"},
    ]

    rendered = format_transcript_with_timestamps(segs)
    ck.check("[00:00-00:02]" in rendered, "转写渲染带 MM:SS 时间戳")
    ck.check("[62:05-62:08]" in rendered or "01:02:05" in rendered,
             "超过 1 分钟/1 小时的时间戳格式正确")

    # 长转写压缩：字数必须降下来，且仍保留时间轴
    long_segs = [
        {"start": i * 2.0, "end": i * 2.0 + 2.0, "text": "这是一段用于测试压缩的长文本内容，第 %d 句。" % i}
        for i in range(600)
    ]
    long_text = " ".join(s["text"] for s in long_segs)
    compressed = compress_segments(long_segs, max_chars=800)
    comp_text = " ".join(s["text"] for s in compressed)
    ck.check(len(comp_text) < len(long_text) * 0.5, f"长转写被压缩（{len(long_text)} -> {len(comp_text)} 字）")
    ck.check(compressed[0]["start"] == long_segs[0]["start"], "压缩后仍保留起始时间戳")
    ck.check(estimate_text_tokens(comp_text) < estimate_text_tokens(long_text), "压缩后文本 token 更少")

    # 进入提示词
    prompt = build_prompt(
        meta={"duration": 60.0, "fps": 25, "has_audio": True, "width": 1280, "height": 720},
        mode="standard", overview_sheets=[], second_sheets=[], transcript=segs,
    )
    ck.contains(prompt, "大家好，今天讲一下视频理解。", "转写内容进入提示词")
    ck.contains(prompt, "音频转写", "提示词中有转写区块")

    # map-reduce 提示词模板
    mp = build_map_prompt({"duration": 600.0, "fps": 25}, 2, 5, 120.0, 240.0)
    ck.contains(mp, "第 2/5 段", "map 提示词含分段序号")
    ck.contains(mp, "02:00-04:00", "map 提示词含该段时间戳")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
CASES = {
    "short": case_short_video,
    "quick": case_quick_and_range,
    "budget": case_budget_guard,
    "long": case_long_video,
    "stt": case_stt_paths,
    "stt-pipeline": case_stt_pipeline,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="video-context-builder 端到端自检")
    ap.add_argument("-k", "--only", default=None, help="只跑指定用例（short/quick/budget/long/stt）")
    ap.add_argument("--keep", action="store_true", help="保留产物目录")
    args = ap.parse_args()

    print("=" * 72)
    print("video-context-builder 自检")
    print("=" * 72)
    print(f"ffmpeg  : {ffmpeg_bin()}")
    print(f"ffprobe : {ffprobe_bin() or '（无，将使用 ffmpeg stderr 兜底解析）'}")
    try:
        import faster_whisper  # noqa: F401
        print("whisper : faster-whisper 已安装")
    except Exception:
        print("whisper : 未安装（STT 用例将验证优雅降级路径）")

    work = Path(tempfile.mkdtemp(prefix="vc_selfcheck_"))
    print(f"工作目录: {work}")

    ck = Checker()
    names = [args.only] if args.only else list(CASES)
    for name in names:
        fn = CASES.get(name)
        if not fn:
            print(f"未知用例：{name}，可选 {list(CASES)}")
            continue
        try:
            fn(ck, work, args.keep) if fn is case_short_video else fn(ck, work)
        except Exception:  # noqa: BLE001
            ck.failed.append(f"用例 {name} 抛异常")
            traceback.print_exc()

    print("\n" + "=" * 72)
    print(f"结果：通过 {len(ck.passed)} 项，失败 {len(ck.failed)} 项")
    if ck.failed:
        for f in ck.failed:
            print("  ✗ " + f)
    print("=" * 72)

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"产物已保留：{work}")
    return 1 if ck.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
