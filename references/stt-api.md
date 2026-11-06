# STT 本地 / API 配置

默认 `local` 使用 faster-whisper；`api` 使用 OpenAI 兼容的同步 `POST /audio/transcriptions`、multipart 文件上传和 Bearer 认证。
API 后端只用 Python 标准库，不需要安装 faster-whisper、torch、OpenAI SDK 或额外 HTTP 库。ffmpeg 仍用于抽音轨。
自托管、无认证的 localhost 服务也支持。其他异步任务、WebSocket、专用签名或不同 JSON 协议需要独立适配，不能只换地址便声称兼容。

## Agent 安装后的配置流程

1. 读取用户指定服务的接口说明，确定 base URL、模型 ID，以及是否支持段时间戳。
2. 在 Skill 根目录复制 `stt.example.json` 为 `stt.local.json`，修改 `backend/api.base_url/api.model/response_format`。
3. 密钥由用户在运行环境或密钥管理器设置。JSON 只记录 `api_key_env` 的变量名，不写密钥值；Agent 不回显密钥，不写入日志或提交。
4. 运行无网络配置检查。通过后，在用户授权使用该服务的任务中选择短音频验证实际服务；检查本身不验证连通性、权限、余额或模型能力。
5. 查看转写状态与时间精度，再让视频理解流程消费材料；失败时保留画面流程，不自动换服务/模型或自动重复上传。

```powershell
# 在 Skill 根目录；编辑复制后的文件，把示例地址与模型换成用户指定值。
Copy-Item -LiteralPath stt.example.json -Destination stt.local.json

# 用户提前在当前 Agent/终端环境中安全设置 STT_API_KEY。
python scripts/transcribe.py --check-config

# 只传配置路径也可，配置文件里的 backend 决定本地或 API。
python scripts/build_video_context.py --video input.mp4 --stt --stt-config stt.local.json --out out.json
```

非 Windows 的复制命令可用 `cp stt.example.json stt.local.json`。
Skill 自带的模板没有真实密钥。`stt.local.json` 和 `.env*` 被 git 忽略；不会自动加载 `.env`，需要宿主/终端注入环境。

## 不建配置文件：环境变量方式

```powershell
$env:STT_BACKEND = "api"
$env:STT_API_BASE_URL = "https://your-provider.example/v1"
$env:STT_API_MODEL = "your-stt-model-id"
$env:STT_API_RESPONSE_FORMAT = "verbose_json"
# STT_API_KEY 的值由用户在运行环境安全配置，示例不包含密钥。
python scripts/transcribe.py --check-config
python scripts/build_video_context.py --video input.mp4 --stt --out out.json
```

Linux/macOS 对应 `export STT_BACKEND=api` 等设置。
若密钥已在其他变量，例如 `MY_ASR_KEY`，设 `STT_API_KEY_ENV=MY_ASR_KEY`；不需要复制密钥。
`--stt-api-key-env MY_ASR_KEY` 也是变量名，不是密钥值。没有传明文密钥的 CLI 参数。

## 配置优先级

CLI / Python 显式参数 → 对应 `STT_*` 环境变量 → JSON → 内置默认。
配置路径：`--stt-config` → `STT_CONFIG_FILE` → Skill 根目录的 `stt.local.json`（存在时）。
示例文件不会自动加载；未配置时 backend 为 local，不会因发现某个 API Key 就自动改用远程服务。

| JSON 字段 | 环境变量 | 命令行 / 默认 |
|---|---|---|
| backend | STT_BACKEND | `--stt-backend`；默认 local |
| api.base_url | STT_API_BASE_URL | `--stt-api-base-url`；API 必填 |
| api.model | STT_API_MODEL | `--stt-api-model`；API 必填，无供应商模型硬编码 |
| api.api_key_env | STT_API_KEY_ENV | `--stt-api-key-env`；默认 STT_API_KEY |
| api.response_format | STT_API_RESPONSE_FORMAT | `--stt-api-response-format`；默认 verbose_json |
| api.timeout | STT_API_TIMEOUT | `--stt-api-timeout`；每次请求默认 120 秒 |
| api.chunk_seconds | STT_API_CHUNK_SECONDS | `--stt-api-chunk-seconds`；默认 300 秒 |
| api.max_upload_mb | STT_API_MAX_UPLOAD_MB | `--stt-api-max-upload-mb`；默认 20 MiB，含 multipart |
| local.model | STT_LOCAL_MODEL | `--whisper-model`；默认 small |
| local.device | STT_LOCAL_DEVICE | `--whisper-device`；默认 auto |
| local.compute_type | STT_LOCAL_COMPUTE_TYPE | `--whisper-compute-type`；默认 int8 |

base_url 通常包含版本路径，如 `https://host.example/v1`，脚本追加 `/audio/transcriptions`；也可填完整转写 endpoint。
禁止 URL 内嵌账号密码/查询密钥。localhost 可不设密钥；远程服务使用配置变量中的 Bearer Key。
配置文件不接受 `api_key` 等明文密钥字段，未知字段会报错以暴露拼写问题。

Python 调用：

```python
from build_video_context import build_video_context

ctx = build_video_context(
    "input.mp4", stt=True, stt_backend="api",
    stt_api_options={
        "base_url": "https://your-provider.example/v1",
        "model": "your-stt-model-id",
        "api_key_env": "STT_API_KEY",
        "response_format": "verbose_json",
    },
)
```

## 时间戳与分块

默认抽为 16kHz 单声道 WAV，按时长和上传字节上限取较小值分块，逐块上传。
指定 `--range` 时仅处理该区间音轨，并把区间偏移、块偏移加回原视频时间轴；不把后续块的 00:00 当成视频开头。
固定分块边界可能切断句子，跨块不做虚构的去重/补词；对白关键时用更小区间复核。

- `verbose_json`：读取 `segments[].start/end/text`；没有段但有 `words` 时保留逐词时间戳。
- `diarized_json`：同样解析 segments，保留服务返回的 speaker 标识，不保证跨音频块的 speaker ID 一致。
- `json / text`：如果只有全文，为每个音频块记录范围并标记 `timestamp_source=chunk`。这不代表逐句对齐；提示词注明精度，也不把块边界作为语音/动作起点。
- 格式由服务及模型决定。verbose_json 不支持时显式改配置，不自动再调用另一个格式，避免重复计费。
- 时间戳缺失、非有限值或明显越界时说明错误，不伪造精确时间。

OpenAI Docs 的 [转写接口](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)
说明了 multipart、段/词时间戳和各模型输出格式差异；选择具体服务/模型时核对其文档。
[文件转写指南](https://developers.openai.com/api/docs/guides/speech-to-text)列明上传大小限制；本工具默认限制为 20 MiB，其他服务按实际限制配置。

## 状态、用量和错误

`transcript_info` 包含 `backend/model/timestamp_precision/api_usage`，不会包含密钥、Authorization 或完整配置。
`api_usage.requests` 为尝试次数，`completed_chunks` 为成功解析块数；分别报告尝试和完成的音频时长，以及服务返回的可识别数值 usage。
STT 的服务计费独立于看图模型 token 预算，失败请求也可能计费；这些字段不是账单或自动费用上限。

中途失败保留已成功的转写，标记 `status=error/partial=true` 并说明其余音频未知。
HTTP 认证、限流、超时、格式错误给出简短提示，不打印可能回显密钥的服务错误正文。
不自动重试、不自动换后端，也不跟随重定向再次发送音频/密钥。

## 验证

```bash
python -m unittest discover -s scripts -p 'test_*.py' -v
python scripts/selfcheck.py
```

API 测试只访问本机模拟服务，验证真实 multipart、认证、分块、时间偏移、异常和主流程集成；不使用真实服务密钥或付费请求。
真实供应商兼容性需在用户配置后以短音频检查；不能用模拟测试声称所有服务均通过。
