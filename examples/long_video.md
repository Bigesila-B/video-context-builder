# 示例二：12 分钟长视频（long-video / map-reduce）

场景：一段 12 分钟的课程录屏，用户问「整堂课讲了哪几个主题？分别在什么时间？」
目标模型只支持图片输入。

**核心问题**：12 分钟 = 720 秒。如果对每一秒都生成秒图（720 张 × 6 帧 = 4320 帧），
token 会直接爆炸（几十万甚至上百万），请求必然失败。
所以长视频必须走「总览 + 关键段 + 分段摘要」的 map-reduce 路线。

---

## 1. 直接跑 auto，策略自动升级

```bash
python scripts/build_video_context.py \
    --video lecture_12min.mp4 \
    --mode auto \
    --max-tokens 50000 \
    --out long.json
```

`auto` 的判定：时长 >= 600 秒（10 分钟）→ 自动切到 `long-video`。

```
[1/6] 元数据: 720.00s, 25.0fps, 1920x1080, 音轨=有
[2/6] 模式: long-video —— >10 分钟：分段 map-reduce，先分段摘要再汇总；STT=开
[4/6] STT 转写中（faster-whisper / small）…
      -> ok (183 段)
[5/6] 采样计划: 总览 32 帧 / 2 张（间隔 22.5s）+ 秒图 16 张（6 帧/张）
      预估 token: 图像 15420 + 文本 3120 = 18540 / 上限 50000
      ! token 估算 18540 <= 上限 50000（占比 37%，帧 128/128）
[6/6] 抽帧与拼图中…
      -> 总览图 2 张（32 帧）
      -> 秒图 16 张
      -> long-video 分段计划：6 段（每段 120s）
完成，用时 42.7s -> long.json
```

**注意几个数字**：

| 指标 | 数值 | 说明 |
|---|---|---|
| 视频时长 | 720s | |
| 理论全量秒图 | 720 张 / 4320 帧 | ❌ 绝不做 |
| 实际总览帧 | 32 帧 / 2 张 | 每 22.5 秒抽 1 帧（原本想 5 秒 1 帧，但 144 帧超预算，自动放宽） |
| 实际秒图 | 16 张 / 96 帧 | 只给「场景切换 + 语音活动 + 均匀补点」 |
| 总帧数 | 128 帧 | 占理论值的 **3%** |
| token | ≈ 18540 | 完全可控 |

---

## 2. 关键秒是怎么挑出来的

`make_second_sheets.pick_key_seconds()` 的优先级：

1. **场景切换**：用 ffmpeg 的 `select='gt(scene,0.3)'` 找镜头切换点 —— 零额外依赖，流式扫描。
2. **语音活动**：从 STT 的 segments 里取每段起始秒 —— 有讲解的地方信息密度高。
3. **均匀补点**：预算还有富余时，按时间等距铺开，保证覆盖面。

三类候选会做「最小间隔 1 秒」去重，最后按优先级截断到预算允许的张数。

`long.json` 里能看到最终选了哪些秒：

```bash
python -c "import json;d=json.load(open('long.json',encoding='utf-8'));print([s['timestamp'] for s in d['second_sheets']])"
# ['00:12', '01:03', '01:47', '02:31', '03:18', '04:02', ...]
```

---

## 3. map-reduce 怎么执行

`long.json` 里多出一个 `map_reduce` 字段：

```jsonc
"map_reduce": {
  "chunk_count": 6,
  "chunk_seconds": 120.0,
  "chunks": [
    {
      "index": 1,
      "start": 0.0, "end": 120.0,
      "start_label": "00:00", "end_label": "02:00",
      "overview_sheets": [ ... ],   // 落在本段的图
      "second_sheets":   [ ... ],   // 落在本段的秒图
      "map_prompt": "你是一个视频理解助手…第 1/6 段…"
    },
    ...
  ],
  "reduce_prompt_template": "你是一个视频理解助手。下面是对同一段视频分段抽样后得到的各段摘要…",
  "how_to": "对每个 chunk：把图 + map_prompt 发给目标模型得到该段摘要；再把所有段摘要填入 reduce_prompt_template 做汇总。"
}
```

执行流程（**这是调用方的事，本工具不调模型**）：

```python
import json, base64, pathlib

ctx = json.loads(pathlib.Path("long.json").read_text(encoding="utf-8"))
mr = ctx["map_reduce"]

def data_url(p):
    return "data:image/jpeg;base64," + base64.b64encode(pathlib.Path(p).read_bytes()).decode()

# ---------- MAP：每段一次请求，单次图片量可控 ----------
summaries = []
for chunk in mr["chunks"]:
    content = [{"type": "text", "text": chunk["map_prompt"]}]
    for s in chunk["overview_sheets"] + chunk["second_sheets"]:
        content.append({"type": "image_url", "image_url": {"url": data_url(s["path"])}})
    summaries.append(call_your_model(content))   # 得到「第 N 段摘要」

# ---------- REDUCE：把摘要拼起来，一次请求汇总 ----------
reduce_prompt = mr["reduce_prompt_template"].replace(
    "<第 i 段摘要>", "\n\n".join(summaries)     # 或者直接把 summaries 拼进 reduce_prompt
)
final = call_your_model([{"type": "text", "text": reduce_prompt}])
print(final)
```

**为什么这样更好**：

- 每段请求只有 ~2 张总览图 + 2-3 张秒图 → 单次调用稳定、不超上下文；
- 汇总阶段只有纯文本 → 便宜；
- 中途某段失败可以只重试那一段，不用整片重跑。

---

## 4. 分段摘要会掉信息吗

会掉一部分，这是取舍。三个缓解手段：

1. **提高预算**：`--max-tokens 120000` → 秒图从 16 张涨到 40+ 张，单段素材更密。
2. **只深挖某段**：先跑 map-reduce 定位到「第 4 段（6:00-8:00）有重点」，再对该段做精细处理：

   ```bash
   python scripts/build_video_context.py --video lecture_12min.mp4 \
       --mode custom-range --range 360-480 --out seg4.json
   ```

   这就是「先粗后细」的两阶段用法，也是最省 token 的组合。

3. **分段再细分**：`map_reduce.chunk_seconds` 可调（默认 120 秒）。改成 60 秒，段数翻倍，
   每段素材更少 → 摘要更细，但请求次数变多。

---

## 5. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 提示「秒图数量已自动下调」 | token 超上限 | 提高 `--max-tokens`，或用 `--range` 聚焦 |
| `mode` 没变成 long-video | 时长 < 600 秒 | 视频确实不够长；可显式 `--mode long-video` |
| 总览间隔被放宽（如 5s → 22.5s） | 总览帧数超预算 | 正常降级，会写进 `warnings` |
| 秒图起点很稀疏 | 场景切换少（比如纯口播） | 增加 STT 语音活动权重，或直接 `--range` 指定 |
| STT 报 SSL / 模型拉取失败 | 本机无法访问 HuggingFace | 见 README「STT 排障」 |
