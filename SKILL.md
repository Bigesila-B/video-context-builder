---
name: video-context-builder
description: >
  把视频转换成「多模态模型可消费的视频上下文包」：按时长与 token 预算自适应采样，视频元数据 + 总览拼图 + 关键片段图
  （时间窗内抽多帧拼成一张图）+ 可选本地/API STT 转写 + 面向目标模型的约束型提示词 + token 估算与超限提醒。
  用于让「只有图片理解能力、没有视频输入能力」的模型间接理解视频内容。
  触发词：视频理解、看懂视频、视频转上下文、视频摘要、秒图、second sheet、关键帧拼图、contact sheet、
  视频抽帧、视频 token 估算、faster-whisper 转写、让模型看视频、video context builder、
  视频分段摘要 map-reduce、视频内容问答。
metadata:
  agent_created: true
---

# Video Context Builder — 让「只能看图」的模型间接看懂视频

## 默认自适应采样（auto / adaptive）

优先使用 `--mode auto`。`adaptive` 是显式同义入口；其余显式模式保留传统策略。
输出的 `mode` 仍按区间落到 standard、long-video 或 custom-range，实际策略看 `sampling_plan.strategy`。
下文传统模式的帧上限、1 秒秒图及 4×4/5×5 布局不适用于自适应策略。

```bash
python scripts/build_video_context.py --video input.mp4 --mode auto --out out.json
python scripts/build_video_context.py --video input.mp4 --mode auto --max-tokens 12000 --question "什么时候拿起钥匙？" --out out.json
python scripts/build_video_context.py --video input.mp4 --mode auto --range 120-180 --max-tokens 8000 --out segment.json
```

- 按有效区间时长选目标密度；预算不足时放宽全程采样间隔，不只保留开头。总览默认 3×2，每图最多 6 格。
- 省略 `--max-tokens` 时，总预算随时长从 6k 逐渐增长至 40k；显式值覆盖自动预算。
- 总预算默认预留 10% 安全余量、15% 后续补看，以及约 10% 回答（256–2048 token）；本轮材料受 `budget.input_limit` 约束。
- 局部图默认覆盖 2 秒、抽 6 帧。各时间分区内优先场景变化、语音和问题匹配的转写关键词。词面匹配不保证语义检索准确度。
- 抽帧后按实际尺寸和 prompt 再核算预算；转写已嵌入 prompt，只计算一次，也不要在发请求时重复附上。
- 完整转写保存在 `transcript_info.full_transcript_path`；仅所选原文片段进入 prompt，保留原始时间戳。
- `budget.ready_to_send=false` 时先调整材料，不自动发送。脚本不调用目标模型。
- `coverage.largest_unsampled_gap_sec` 报告实际采样时间点最大空白，含首尾边界；逐帧真实时间看 `frame_timestamps`。
- 补看由外层 Agent 使用 `--mode auto --range a-b` 调度，累计用量由调用方记录，不能每轮重置预算后宣称仍在同一总额内。
- 主包预算针对一次请求；map/reduce 是替代路线，须另查累计 map 输入和输出/reduce 成本，不能默认两条路线都调用。

时长档位与预算细节见 [自适应采样](references/adaptive-sampling.md)。
STT 支持本地 faster-whisper 和 OpenAI 兼容 API；模型选型见 [STT 建议](references/stt-models.md)。

## 安装后配置 STT API

用户选择 API 时，先读 [API 配置说明](references/stt-api.md)，核对所选服务的 endpoint、模型和返回格式。
将 Skill 根目录 `stt.example.json` 复制为 `stt.local.json`，配置 `backend=api`、地址、模型与 `api_key_env`。
密钥由用户在运行环境/密钥管理器提供；Agent 只检查存在性，不回显，不把值写入配置、对话日志或提交。
配置优先级为显式参数 → STT_* 环境变量 → JSON → 默认本地后端；配置模板不自动生效。

```bash
python scripts/transcribe.py --check-config --stt-config stt.local.json
python scripts/build_video_context.py --video input.mp4 --stt --stt-config stt.local.json --out out.json
```

配置检查不联网、不加载模型，不代表远程模型/余额/权限已验证。API 模式会向用户配置的服务上传音频；不因本地失败自动改用远程服务。
API 不依赖 faster-whisper。请求失败不自动重试或换模型，避免重复计费；记录 `transcript_info.api_usage`。
只有全文的 API 标注为音频块级范围，不能用块边界当作逐句/动作时间。时间戳格式和供应商差异见上述说明。
`--range` 仅转写该区间并保留原片时间偏移。需要完全离线时显式 `--stt-backend local`。

修改后除本文原有自检，还运行：

```bash
python -m unittest discover -s scripts -p test_adaptive_sampling.py -v
python -m unittest discover -s scripts -p test_stt_api.py -v
```

以下表格说明传统固定模式。一般任务采用上面的自适应入口。

## 这个 Skill 解决什么问题

目标多模态模型**不支持视频输入**，只吃图片和文本。
本 Skill 把视频切成「模型能消费的最小充分材料」：

| 层 | 产出 | 作用 |
|---|---|---|
| 总览层 | 每 2–5 秒抽 1 帧，拼成 4x4 / 5x5 总览图 | 快速**定位**——视频大致讲了什么、结构如何 |
| 关键秒层 | 1 秒内抽多帧（默认 6 帧）拼成 3x2 的**秒图** | 看清**细节**——动作分解、瞬间表情、画面变化 |
| 音频层 | faster-whisper 转写，带 start/end 时间戳 | 补上画面之外的信息 |
| 提示层 | 约束型提示词 | 禁止模型编造，强制引用时间戳 |
| 预算层 | token 估算 + 超限提醒 + 自动降级 | 防止静默把几百张图丢给模型 |

**核心原则：不要对整段长视频全量生成秒图。** 长视频只对「关键时间段」出秒图。

## 何时使用

- 用户给你一段视频（或视频路径），问「讲了什么 / 找出某个片段 / 为什么 XX 发生了」
- 用户想用某个**只支持图片**的模型（Qwen-VL、GPT-4o、GLM-4V、Claude 等）分析视频
- 用户需要视频的文本化摘要、时间轴笔记、内容审核、运营素材拆解

## 前置检查（Agent 必做）

```bash
# 1) ffmpeg 是否可用（没有系统 ffmpeg 也没关系，装 imageio-ffmpeg 即可）
python -c "import sys; sys.path.insert(0,'scripts'); from ffmpeg_utils import ffmpeg_bin; print(ffmpeg_bin())"

# 2) 仅在缺依赖时补齐最低依赖
pip install Pillow imageio-ffmpeg
```

若报 `FFmpegNotFoundError`，执行 `pip install imageio-ffmpeg` 即可（自带二进制，无需管理员权限）。
本地 STT 需要 `pip install faster-whisper`；未安装时该后端会跳过转写并说明原因。API 后端不需要此包。

## 用法一：CLI

```bash
# 自动模式：按有效区间时长和预算安排密度
python scripts/build_video_context.py --video input.mp4 --mode auto --stt \
    --max-tokens 50000 --out out.json

# 只看 2:00-3:00，逐秒出秒图
python scripts/build_video_context.py --video input.mp4 \
    --mode custom-range --range 120-180 --out seg.json

# 超省 token：只要总览
python scripts/build_video_context.py --video input.mp4 --mode quick \
    --max-tokens 8000 --out quick.json

# 精度优先，并声明目标模型（影响 token 估算口径）
python scripts/build_video_context.py --video input.mp4 --mode deep \
    --model-preset qwen-vl --out deep.json
```

## 用法二：Python 函数

```python
import sys; sys.path.insert(0, "scripts")
from build_video_context import build_video_context

ctx = build_video_context(
    "input.mp4",
    mode="auto",          # auto / quick / standard / deep / long-video / custom-range
    stt=True,
    max_tokens=50000,
    custom_range=None,    # 或 (120, 180) / "120-180"
)

print(ctx["token_estimate"])       # {'images': ..., 'text': ..., 'total': ...}
print(ctx["overview_sheets"])      # 总览图清单
print(ctx["second_sheets"])        # 局部图清单（时长以 start/end 为准）
print(ctx["prompt"])               # 直接发给目标模型的提示词
```

## 发给目标模型时怎么组装

**只发图片和文本，不要提「视频」二字之外的任何隐含输入。**

```python
content = [{"type": "text", "text": ctx["prompt"]}]
# 顺序必须与 prompt 中的「材料清单」一致：总览图 → 秒图 → 原图
for s in ctx["overview_sheets"] + ctx["second_sheets"] + ctx.get("original_frames", []):
    content.append({"type": "image_url", "image_url": {"url": file_to_data_url(s["path"])}})
```

## 六种模式怎么选

| 模式 | 触发条件 | 总览 | 秒图 | STT | 帧上限 |
|---|---|---|---|---|---|
| `quick` | 只想知道大致内容 | 1–2 张 | 无 | 关 | 32 |
| `standard` | **默认**，大多数场景 | 有 | 关键秒 | 开 | 96 |
| `deep` | 需要看清细节/精确时间点 | 有 | 关键段 | 开 | 200 |
| `long-video` | 时长 > 10 分钟 | 有 | 关键段 | 开 | 128 |
| `custom-range` | 用户指定时间段 | 区间内 | 区间内逐秒 | 开 | 112 |
| `auto` / `adaptive` | 默认 | 按有效时长和预算规划，总览 3×2 | 关键片段 2 秒窗 | 开 | 动态 |

## token 控制要点（重要）

1. 处理前会**先估算**图像 token 与文本 token，再决定抽多少帧——不是先抽完再说。
2. 超过 `--max-tokens` 时：
   - 输出 JSON 的 `budget.over_budget` 为 `true`，`warnings` 里写明超了多少；
   - 默认**自动下调局部图数量**（全程分散保留时间点），并明确告知降了多少；
   - 用 `--no-degrade` 可改为「只提醒不动手」。
3. 最省 token 的三种手段：`--mode quick`、`--range a-b`、调小 `--cell` / 调大 `--overview-interval`。

## 采样密度怎么算（常被问）

两层采样，密度不同，**都在 JSON 的 `sampling_plan` 里可查**：

| 层 | 密度 | 作用范围 |
|---|---|---|
| 总览层 | 每 `overview_interval` 秒 1 帧（默认 2–5s，即 0.2–0.5 帧/秒） | 覆盖全片 |
| 秒图层 | 每张覆盖 1 秒、抽 `second_frames` 帧（默认 6 → 6 帧/秒） | 只覆盖被选中的关键秒 |

⚠ **陷阱**：`--overview-interval 1` 不一定会得到 1 帧/秒。每个模式都有总览帧数上限
（`overview_cap`：quick/standard/long-video 32、deep 50），超了会自动**放宽间隔**并在 `warnings` 里说明。
要真按固定间隔抽，必须同时给 `--overview-cap`：

```bash
# 62 秒视频，要每 1 秒 1 帧 → cap 必须 >= 63
python scripts/build_video_context.py --video in.mp4 --mode deep \
    --overview-interval 1.0 --overview-cap 70 --out out.json
```

想看某个参数的最终实际取值，读 `out.json` 的 `sampling_plan.overview_interval` 与
`sampling_plan.overview_frame_count`，不要只信命令行。

## 长视频（至少 10 分钟）的可选 map-reduce 路线

`mode=long-video` 时输出 JSON 里会多出 `map_reduce` 字段：

```text
map_reduce.chunks[i] = {
  index, start, end, start_label, end_label,
  overview_sheets: [...],   # 本段的图
  second_sheets:   [...],   # 本段的秒图
  map_prompt:      "..."    # 发给模型做「本段摘要」的提示词
}
map_reduce.reduce_prompt_template  # 把各段摘要填进去，做最终汇总
```

执行顺序：**逐段 map（每段一次调用）→ 收集摘要 → reduce（一次调用）**。
这样可拆分单次请求的图片量；累计 token 仍需另算，检查 estimated_map_input_tokens 并计入各段输出及 reduce 成本。

## 输出 JSON 结构

```jsonc
{
  "meta": {"duration": 123.4, "fps": 30, "has_audio": true, "width": 1920, "height": 1080},
  "mode": "standard",
  "overview_sheets": [{"path": "...", "start": 0, "end": 60, "layout": "4x4", "cells": 16}],
  "second_sheets":   [{"path": "...", "start": 12, "end": 13, "frames": 6, "layout": "3x2",
                       "timestamp": "00:12", "frame_timestamps": [12.0, 12.17, ...]}],
  "original_frames": [],          // 仅 deep 模式
  "transcript": [{"start": 0.0, "end": 2.3, "text": "..."}],
  "transcript_info": {"status": "ok|skipped|unavailable|error", "reason": "..."},
  "prompt": "...",
  "token_estimate": {"images": 12000, "text": 3000, "total": 15000},
  "token_breakdown": {"overview": ..., "second": ..., "originals": ..., "prompt": ..., "transcript": ...},
  "budget": {"max_tokens": 50000, "over_budget": false, "ratio": 0.3, "message": "..."},
  "sampling_plan": {"overview_interval": 3.0, "frames_total": 92, "degraded": false, ...},
  "warnings": ["..."],
  "map_reduce": {...}             // 仅 long-video 模式
}
```

## 需要记住的行为约定

- **时间戳精确到秒**：秒图的 `timestamp` 用 `MM:SS` / `HH:MM:SS`，`frame_timestamps` 保留浮点。
- **无音轨自动跳过 STT**：`transcript_info.status == "skipped"`，不会抛异常。
- **不整段载入内存**：全流程 ffmpeg 流式解码，抽出的帧落盘到临时目录，收尾自动清理。
- **临时文件**：默认清理；`--keep-temp` 可保留以便排查。
- **视频理解调用边界**：本工具不调用看图/视频理解模型，只准备材料；已配置并选用 API STT 时会上传音频进行转写。

## 自检（改完代码务必跑一遍）

```bash
python scripts/selfcheck.py            # 自动生成测试视频并验证全部验收标准
python scripts/selfcheck.py --keep     # 保留产物便于肉眼检查
```

自检覆盖：30 秒视频出总览图 + 秒图 + prompt；10 分钟视频不生成 600 张秒图；
时间戳准确到秒；无音轨自动跳过 STT；超预算提醒与降级；临时目录清理干净。

## 常见坑与排障

| 现象 | 原因 | 处理 |
|---|---|---|
| `FFmpegNotFoundError` | 系统没装 ffmpeg | `pip install imageio-ffmpeg`（自带二进制，免管理员） |
| 拼图里时间戳与画面错位约 2 帧 | 用朴素 `-ss`/`fps` 抽帧会踩 B 帧 DTS 延迟与 fps 取帧偏移 | 本工具已用 `trim + select + showinfo + -fps_mode passthrough` 修掉，**不要退回 `-vf fps=N`** |
| `transcript_info.status == "unavailable"` | 未装 faster-whisper | `pip install faster-whisper`，或忽略（会自动跳过） |
| `transcript_info.status == "error"` 且提到 SSL / Hub | 首次使用需从 HuggingFace 下载模型，受限网络下失败 | 设 `HF_ENDPOINT=https://hf-mirror.com`；或配代理根证书（`SSL_CERT_FILE`）；或预下模型后 `--whisper-model <本地目录>` |
| token 超限提醒 | 抽样量超预算 | 输出里已给建议：改 quick / 用 `--range` / 调小 `--cell` / 调大 `--overview-interval` |
| 拼图有大量黑边空白 | 帧数不满整行整列 | 已处理（画布自动收缩到实际行数）；若仍异常请检查 `--cell` 与源宽高比 |
| 长视频跑得慢 | 关键段检测需要全片扫描一次 | 正常；用 `--range` 可只扫一段 |

## 验证方式（改完代码必跑）

`selfcheck.py` 用「每帧内容编码了秒号与帧号」的合成视频做**像素级反查**：
从生成的拼图里读回像素 → 复原「第几秒第几帧」→ 与 JSON 里声明的时间戳逐一比对。
这是「时间戳准确到秒」的可自动复现证据，不依赖人眼。

```bash
python scripts/selfcheck.py            # 6 个用例 / 57 项断言
python scripts/selfcheck.py -k short   # 只跑某一类（short/quick/budget/long/stt/stt-pipeline）
python scripts/selfcheck.py --keep     # 保留产物便于目检
```

## 文件结构

```
video-context-builder/
  SKILL.md
  README.md
  requirements.txt
  scripts/
    ffmpeg_utils.py          # ffmpeg/ffprobe 定位与调用（含 imageio-ffmpeg 兜底）
    probe_video.py           # 元数据探测（ffprobe 优先，ffmpeg stderr 兜底）
    extract_frames.py        # 帧精确抽帧：等间隔 / 时间窗密集 / 单帧原图
    make_contact_sheet.py    # 拼图引擎（网格 + 时间戳标签 + 自动收缩空行）
    make_second_sheets.py    # 关键段检测（场景/语音/补点）+ 秒图生成
    transcribe.py            # STT（faster-whisper，可选，优雅降级）
    token_estimator.py       # token 估算与预算检查
    prompt_builder.py        # 提示词构造（含 map/reduce 模板）
    build_video_context.py   # 主编排器 + CLI
    selfcheck.py             # 端到端自检（像素级验证时间戳）
  examples/
    short_video.md           # 30 秒短视频 walkthrough
    long_video.md            # 12 分钟长视频 map-reduce walkthrough
    demo_output/             # 真实跑出来的产物（demo_clip.mp4 + 拼图 + JSON）
```
