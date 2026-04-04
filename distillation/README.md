# distillation — SVG 蒸馏模块

从种子 query（`high_priority_pool.jsonl` 或 `anneal_pool.jsonl`）中批量调用 LLM API，
生成 SVG 代码，写出蒸馏结果。

---

## 目录结构

```
distillation/
  config.py          # DistillConfig dataclass + YAML 加载
  prompt.py          # SVG system prompt + user content 构造
  distill.py         # run_distill()：并行调用 + resume + 流式写出
  demo.py            # App 1：单条调用 debug（直接改参数运行）
  main.py            # CLI 入口
  configs/
    default.yaml     # 配置模板

utils/
  api_client.py      # 通用 OpenAI-compatible 调用器（本模块依赖）
```

---

## 快速开始

### 安装依赖

```bash
pip install requests pyyaml loguru
```

### App 1：单条 debug（验证 API 连通性 / prompt 效果）

编辑 `distillation/demo.py` 顶部的参数：

```python
URL     = "http://localhost:8000/v1/chat/completions"
API_KEY = "your-api-key"
MODEL   = "your-model-name"
TEXT    = "Draw a simple red circle with a blue border"
IMAGE   = None   # 图文请求时填本地图片路径
```

运行：

```bash
python -m distillation.demo
```

### App 2：批量蒸馏

编辑 `distillation/configs/default.yaml`，至少填写：

```yaml
url: "http://localhost:8000/v1/chat/completions"
api_key: "your-api-key"
model: "your-model-name"
input_path: "/path/to/high_priority_pool.jsonl"
output_path: "/path/to/output/svg_responses.jsonl"
```

运行：

```bash
# 全量
python -m distillation.main --config distillation/configs/default.yaml

# 调试：只跑前 10 条
python -m distillation.main --config distillation/configs/default.yaml --limit 10

# 从头重跑（忽略已有输出）
python -m distillation.main --config distillation/configs/default.yaml --no-resume
```

---

## 配置参考

```yaml
# 完整 endpoint URL（必须包含路径）
# 本地 vLLM：http://localhost:8000/v1/chat/completions
# OpenAI：   https://api.openai.com/v1/chat/completions
url: "http://localhost:8000/v1/chat/completions"
api_key: "your-api-key"
model: "your-model-name"

timeout: 120.0       # 单次请求超时秒数
max_retries: 3       # 可重试错误的最大重试次数
ssl_verify: true     # 本地 vLLM 自签名证书时改为 false
log_user: "svg_distill"

input_path: "/path/to/high_priority_pool.jsonl"
output_path: "/path/to/svg_responses.jsonl"
call_log_path: "logs/api_calls.jsonl"   # 每次调用明细

num_workers: 16    # 并发线程数（推荐 16–32）
resume: true       # 断点续跑
```

---

## 输出格式

### 蒸馏结果（`output_path`）

每行一条 JSON：

```json
{"id": "stage1_icon/text2svg/data_000000:312",
 "instruction": "Draw an orange circle...",
 "status": "ok",
 "response": "<svg xmlns=...>...</svg>",
 "error": null}
```

失败时：

```json
{"id": "...",
 "instruction": "...",
 "status": "error",
 "response": null,
 "error": "HTTPError: 500 Internal Server Error"}
```

### API 调用明细（`call_log_path`）

每次调用追加一条（成功或失败均记录）：

```json
{"ts": "2026-04-04T10:00:00+00:00", "url": "http://...",
 "model": "your-model", "status": "ok",
 "prompt_tokens": 42, "completion_tokens": 512, "error": null}
```

---

## Retry 策略

| 错误类型 | 行为 |
|---------|------|
| HTTP 429 | 读 `Retry-After` 头，无则指数退避（1s → 2s → 4s） |
| HTTP 5xx（500/502/503/504） | 指数退避重试 |
| HTTP 400/401/403/404 | **不重试**，立即失败（参数错误） |
| Timeout / ConnectionError | 指数退避重试 |

超过 `max_retries` 后，该条记录写出 `status=error`，继续处理其余记录。

---

## Resume 机制

启动时扫描已有 `output_path`，收集所有 `id` 字段，跳过已完成的记录。
中途 Ctrl+C 或崩溃后，重跑同一命令即可从断点继续，不会重复调用。

---

## 图文请求（App 1）

`demo.py` 支持发送图片 + 文字：

```python
IMAGE = "/path/to/reference.png"
TEXT  = "Generate an SVG similar to this image"
```

图片会被 base64 编码，以 `data:image/png;base64,...` 格式内联在请求中。
批量蒸馏（App 2）当前仅使用文本 instruction，图文支持可在 `prompt.py` 中扩展。

---

## 通用 API 调用器（`utils/api_client.py`）

若要在其他任务中直接调用 API，可复用：

```python
from utils.api_client import call_chat_completion, text_content, image_text_content

result = call_chat_completion(
    url="http://host/v1/chat/completions",
    api_key="token",
    model="model",
    user_content=text_content("Hello"),
    system="You are a helpful assistant.",
)
```
