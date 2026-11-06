# STT 模型与运行策略

当前支持本地 faster-whisper，以及可配置的 OpenAI 兼容转写 API，配置步骤见 [STT API](stt-api.md)。
以下本地模型选型没有经过本仓库同机准确率评测；SenseVoice 原生本地运行时尚未接入。

| 场景 | 候选 | 取舍 |
|---|---|---|
| CPU 粗读、低资源 | faster-whisper base + int8 | 易与现有流程兼容；难辨对白需要复核 |
| 通用 CPU 平衡 | faster-whisper small + int8 | 保持当前默认，避免未经实测降低转写质量 |
| 中文/粤语优先 | SenseVoiceSmall，可研究 ONNX/量化运行时 | 中、粤、英、日、韩；情绪/事件标签可补充视频线索，时间戳方案需单独验证 |
| GPU 多语言精读 | faster-whisper large-v3-turbo | 官方模型裁剪解码层以提高速度；资源需求仍需实测 |
| 英文为主 | distil-large-v3 | 官方为英文系列，不能默认推荐给中文视频 |

来源（查阅于 2026-11-06）：

- [faster-whisper 官方 README](https://github.com/SYSTRAN/faster-whisper)：CTranslate2、CPU/GPU int8、VAD、推理示例和同机基准。
- [SenseVoiceSmall 官方模型卡](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)：小模型的五种语言、情绪与声音事件能力。不要将整个系列的语言数当作 Small 的语言数。
- [sherpa-onnx 官方仓库](https://github.com/k2-fsa/sherpa-onnx)：研究离线 ONNX 部署时的候选运行时，接入前检查具体模型/量化文件与时间戳输出。
- [Whisper large-v3-turbo 官方模型卡](https://huggingface.co/openai/whisper-large-v3-turbo)：解码层由 32 减至 4，提速伴随质量取舍。
- [Distil-Whisper 官方模型卡](https://huggingface.co/distil-whisper/distil-large-v3)：英文 distil-large-v3。

## 推荐的后续优化

优先用已有字幕轨/用户提供的字幕作为可选材料，并注明来源；字幕本身也可能不准确。
无字幕时 VAD 过滤静音；粗读采用轻模型；用户追问或识别异常时，仅对相关音频窗用较强模型重跑。
缓存需包含音频指纹、模型/版本、语言、精度和解码配置，不能只按文件名命中。
音频剪段后要把原视频偏移加回时间戳，不能把段内 00:00 误当视频开头。

保持完整转写用于回查，送模型的是原文证据的预算选择，不用“最长一句”冒充完整内容。
STT 本地推理节省的是时间/内存；语言模型的文本 token 需要独立预算。
不同运行时的模型权重格式不通用，不能把任意 Transformers 目录直接当作 CTranslate2 模型目录。

接入 SenseVoice 时应单独保留 `emotion/audio_events`，标记为模型推断；不要把标签拼进用户对白原文。
需要精确字词对齐时评估额外对齐器；VAD 切段范围只表示有声片段边界。
