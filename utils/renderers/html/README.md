# HTML 渲染管线

基于 Playwright (Chromium) 的批量 HTML 渲染管线，包含三个阶段：

| 阶段 | 脚本 | 作用 |
|------|------|------|
| 1. 渲染 | `render_html.py` | 从 JSON 数组读取 `html` 字段，全页截图为 PNG |
| 2. 解析 | `parse_render_log.py` | 解析渲染日志，提取 WARN/ERROR，生成 `issues.json` |
| 3. 过滤 | `filter_by_issues.py` | 根据 `issues.json` 从原数据中剔除有问题的 id |

`html_render.py` 是统一入口，通过子命令 `render / parse / filter` 分发到上述三个脚本。

---

## 快速上手

```bash
# 1. 渲染（stderr 重定向到日志文件）
python html_render.py render \
    --json_dir /data/json \
    --files part_001.json part_002.json \
    --images_dir /data/images \
    --workers 16 \
    2>render.log

# 2. 解析日志，生成问题 id 报告
python html_render.py parse render.log -o issues.json

# 3. 过滤原数据，输出干净副本（仅剔除有 ERROR 的 id）
python html_render.py filter \
    --issues issues.json \
    --input /data/json \
    --output /data/json_clean \
    --level error
```

每个子命令都支持 `--help` 查看完整参数：

```bash
python html_render.py render --help
python html_render.py parse --help
python html_render.py filter --help
```

也可以直接运行各脚本（参数相同）：

```bash
python render_html.py --json_dir ... --files ... --images_dir ...
python parse_render_log.py render.log -o issues.json
python filter_by_issues.py --issues issues.json --input data/ --output clean/
```

---

## 输出结构

```
images_dir/
  {json_stem}/
    {id}.png          # 渲染截图（幂等，已存在则跳过）

_cdn_cache/           # CDN 离线缓存（默认在 images_dir/../_cdn_cache）
  <sha256>.<ext>
  <sha256>.<ext>.meta

render.log            # 渲染期 stderr，结构化日志
issues.json           # parse 阶段生成的问题 id 报告

json_clean/           # filter 阶段输出的干净数据副本
  part_001.json
```

---

## 日志类别

渲染期所有异常均写入 stderr，格式：

```
[LEVEL] | [CATEGORY] | id=file:id | key=value | msg=...
```

| 类别 | 级别 | 含义 |
|------|------|------|
| `INFINITE_GROWTH` | WARN | 页面高度持续增长，可能渲染出极长的图 |
| `TEXT_OVERLAP` | WARN | 检测到文字元素相互叠压 |
| `TIMEOUT` | ERROR | goto 或 screenshot 超时 |
| `WAIT_TIMEOUT` | WARN | networkidle 等待超时（已继续截图） |
| `WAIT_FINAL_TIMEOUT` | WARN | 两轮等待均超时，强制截图 |
| `REQUEST_FAILED` | ERROR | 网络请求失败 |
| `HTTP_ERROR` | ERROR | HTTP 4xx/5xx 响应 |
| `PAGE_ERROR` | ERROR | 页面 JS 运行时错误 |
| `CONSOLE_ERROR` | WARN | 控制台 error 输出 |
| `EMPTY_OUTPUT` | ERROR | 截图文件缺失或小于 1KB |
| `BAD_DATA` | ERROR | html 字段为空 |
| `RENDER_ERROR` | ERROR | 其他渲染异常 |

---

## render 子命令参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--json_dir` | 必填 | JSON 文件所在目录 |
| `--files` | 必填 | JSON 文件名列表（空格分隔） |
| `--images_dir` | 必填 | 输出图片根目录 |
| `--workers` | `cpu_count()-1` | 并行进程数 |
| `--cache_dir` | `<images_dir>/../_cdn_cache` | CDN 缓存目录 |
| `--cache_concurrency` | 16 | 预下载并发线程数 |
| `--skip_cache_download` | False | 跳过预下载（仅用已有缓存） |
| `--no_cache` | False | 完全禁用 CDN 缓存 |
| `--base_max_wait` | 6000 | 单次 networkidle 等待基础上限（ms） |
| `--page_total_timeout` | 60000 | 单页总超时硬上限（ms） |
| `--recycle_every` | 200 | 每渲染 N 张回收一次浏览器 |
| `--no_growth_check` | False | 关闭无限拉长检测 |
| `--growth_min_window_ms` | 1000 | 拉长判定的最小窗口时长（ms） |
| `--growth_threshold_px` | 500 | 窗口内净增长阈值（px） |
| `--no_overlap_check` | False | 关闭文字叠压检测 |
| `--overlap_ratio_threshold` | 0.20 | 叠压面积比阈值 |

## parse 子命令参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `logs` (位置参数) | 必填 | 日志文件路径，支持多个或通配符 |
| `-o / --output` | `render_issues.json` | 输出 JSON 路径 |

## filter 子命令参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--issues` | 必填 | parse 生成的 issues.json 路径 |
| `--input` | 必填 | 原数据 JSON 文件或目录 |
| `--output` | 必填 | 过滤后副本的输出目录 |
| `--level` | `all` | `all`=ERROR+WARN 都剔除；`error`=仅剔除有 ERROR 的 id；`warn`=仅剔除有 WARN 的 id |

---

## 各模块说明

### render_html.py

- **CDN 离线缓存**：渲染前扫描所有 HTML，预下载 CDN 资源到本地；渲染时通过 Playwright 路由拦截，命中缓存直接 fulfill，避免反复拉网络。
- **图表动画禁用**：注入 JS 脚本，在 `window.Chart` / `window.echarts` 赋值时即时关闭动画，确保截图时图表已绘制完毕。
- **无限拉长检测**：注入高度采样器（200ms/次），在等待和截图前用滑动窗口判定页面高度是否持续增长；命中时记录 `WARN INFINITE_GROWTH`，仍然截图（由下游根据日志决定是否丢弃）。
- **文字叠压检测**：截图前注入 JS，枚举可见文本节点，两两检测 BoundingClientRect 是否显著重叠（排除祖先-后代关系）；命中时记录 `WARN TEXT_OVERLAP`。
- **系统负载自适应**：读取 `os.getloadavg()`，负载高时自动放大等待超时，避免在繁忙机器上误判。
- **断点续跑**：扫描已存在的 `*.png`，跳过已完成的 id，支持中断后重启。
- **浏览器定期回收**：每渲染 `recycle_every` 张关闭并重启浏览器，防止内存/句柄泄漏。

### parse_render_log.py

解析 `render_html.py` 写到 stderr 的结构化日志，按 `log_id`（`文件名:item_id`）归组，输出 JSON 报告，包含：

- `summary`：总计 WARN/ERROR 条目数、各类别分布、涉及 id 数量
- `ids`：每个 id 的所有问题条目，含 `has_error` / `has_warn` 标志

### filter_by_issues.py

读取 `issues.json`，从原数据 JSON 数组中剔除指定 id 的条目，写到输出目录（原文件不变）。自动探测原文件的编码、BOM、缩进格式，输出保持一致。
