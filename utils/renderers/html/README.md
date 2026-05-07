# HTML 渲染管线

基于 Playwright (Chromium) 的批量 HTML 渲染管线，包含四个阶段：

| 阶段 | 脚本 | 作用 |
|------|------|------|
| 1. 渲染 | `render_html.py` | 从 JSON 数组读取 `html` 字段，全页截图为 PNG |
| 2. 解析 | `parse_render_log.py` | 解析渲染日志，提取 WARN/ERROR，生成 `issues.json` |
| 3. 过滤 | `filter_by_issues.py` | 根据 `issues.json` 从原数据中剔除有问题的 id |
| 4. 统计 | run.sh 内嵌 | 扫描 PNG 尺寸分布，生成统计图表 |

`html_render.py` 是统一入口，通过子命令 `render / parse / filter` 分发到上述脚本。

---

## 快速上手

```bash
# 一键运行全部阶段（默认 resume 模式）
bash run.sh

# 常用选项
bash run.sh --workers 8 --level error
bash run.sh --tall_ratio 3.0        # 高宽比阈值（默认 4.0）
bash run.sh --no-resume              # 清空日志，重新记录（已有 PNG 不删除）
```

手动逐步运行：

```bash
# 1. 渲染（stderr 重定向到日志文件，追加模式）
python html_render.py render \
    --json_dir /data/json \
    --files part_001.json part_002.json \
    --images_dir /data/images \
    --workers 16 \
    --tall_ratio_threshold 4.0 \
    2>>render.log

# 2. 解析日志，生成问题 id 报告
python html_render.py parse render.log -o issues.json

# 3. 过滤原数据，输出干净副本（仅剔除有 ERROR 的 id）
python html_render.py filter \
    --issues issues.json \
    --input /data/json \
    --output /data/json_clean \
    --level error \
    --stats_dir /data/stats        # 可选：生成过滤统计图表
```

每个子命令都支持 `--help` 查看完整参数：

```bash
python html_render.py render --help
python html_render.py parse --help
python html_render.py filter --help
```

---

## Resume 机制

渲染阶段默认开启 resume：

- **render**：已存在的 `{id}.png` 自动跳过，不重复渲染。
- **日志**：以追加模式（`>>`）写入 `render.log`，历史 WARN/ERROR 不会丢失。
- parse 和 filter 每次都全量重跑，基于累积的完整日志产生正确统计。

使用 `--no-resume` 可清空日志重新开始（已有 PNG 不会被删除，如需重新渲染需手动删除）。

---

## 输出结构

```
images_dir/
  {json_stem}/
    {id}.png          # 渲染截图（幂等，已存在则跳过）

_cdn_cache/           # CDN 离线缓存（默认在 images_dir/../_cdn_cache）
  <sha256>.<ext>
  <sha256>.<ext>.meta

render.log            # 渲染期 stderr，结构化日志（追加模式）
issues.json           # parse 阶段生成的问题 id 报告

filtered/             # filter 阶段输出的干净数据副本
  {sub_dir}/
    output.jsonl

stats/                # 统计图表
  drop_ratio.png          # 各文件过滤比例
  {sub}_filter.json       # 机器可读的过滤汇总（用于全局聚合）
  image_dimensions.png    # 所有 PNG 的宽度/高度分布
  image_ratio.png         # 所有 PNG 的高宽比分布
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
| `REQUEST_FAILED` | ERROR/WARN | 网络请求失败（字体/装饰性 CDN 降为 WARN） |
| `HTTP_ERROR` | ERROR/WARN | HTTP 4xx/5xx 响应（字体/装饰性 CDN 降为 WARN） |
| `PAGE_ERROR` | ERROR | 页面 JS 运行时错误 |
| `CONSOLE_ERROR` | WARN | 控制台 error 输出 |
| `EMPTY_OUTPUT` | ERROR | 截图文件缺失或小于 1KB |
| `BAD_DATA` | ERROR | html 字段为空 |
| `RENDER_ERROR` | ERROR | 其他渲染异常 |
| `TALL_PAGE` | WARN | 截图高宽比超过阈值 |

> **字体 CDN 降级**：`fonts.googleapis.com`、`fonts.gstatic.com`、FontAwesome 等装饰性资源加载失败不影响核心渲染质量，统一降为 WARN，避免大量字体网络超时污染过滤结果。

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
| `--tall_ratio_threshold` | 3.0 | 高宽比超过此值记录 WARN（0 禁用），run.sh 默认 4.0 |

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
| `--level` | `all` | `all`=ERROR+WARN 都剔除；`error`=仅剔除有 ERROR；`warn`=仅剔除有 WARN |
| `--summary_json` | 无 | 将本次统计写入 JSON 文件（供外层脚本聚合） |
| `--stats_dir` | 无 | 生成过滤比例图表并保存到此目录（需要 matplotlib） |

---

## run.sh 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--workers N` | `nproc-1` | 并行进程数 |
| `--level L` | `all` | 过滤级别（error/warn/all） |
| `--tall_ratio F` | `4.0` | 高宽比 WARN 阈值 |
| `--no-resume` | — | 清空日志重新开始（默认追加续跑） |

环境变量也可覆盖：`WORKERS`、`FILTER_LEVEL`、`TALL_RATIO_THRESHOLD`、`RESUME`。

---

## 各模块说明

### render_html.py

- **CDN 离线缓存**：渲染前扫描所有 HTML，预下载 CDN 资源到本地；渲染时通过 Playwright 路由拦截，命中缓存直接 fulfill，避免反复拉网络。
- **字体 CDN 鲁棒性**：Google Fonts 等装饰性 CDN 的请求失败降为 WARN 而非 ERROR，避免字体网络超时导致大量记录被过滤。
- **图表动画禁用**：注入 JS 脚本，在 `window.Chart` / `window.echarts` 赋值时即时关闭动画，确保截图时图表已绘制完毕。
- **无限拉长检测**：注入高度采样器（200ms/次），用滑动窗口判定页面高度是否持续增长；命中时记录 `WARN INFINITE_GROWTH`。
- **文字叠压检测**：截图前注入 JS，枚举可见文本节点，两两检测 BoundingClientRect 是否显著重叠。
- **高宽比检测**：截图后检测 `高度/宽度 > tall_ratio_threshold`，命中时记录 `WARN TALL_PAGE`。
- **系统负载自适应**：读取 `os.getloadavg()`，负载高时自动放大等待超时。
- **断点续跑**：扫描已存在的 `*.png`，跳过已完成的 id，支持中断后重启。

### parse_render_log.py

解析 `render_html.py` 写到 stderr 的结构化日志，按 `log_id` 归组，输出 JSON 报告。

### filter_by_issues.py

读取 `issues.json`，从原数据中剔除指定 id，写到输出目录（原文件不变）。输出格式：
- 每个输入文件：`filename  剔除 X/Y (Z%)  保留 N`（直通文件标记 `[直通]`）
- SUMMARY：仅统计实际参与过滤的文件，直通文件单独列出，不混入总数
- `--summary_json`：写出机器可读的统计 JSON，供 run.sh 跨子目录聚合
- `--stats_dir`：生成 `drop_ratio.png` 柱状图（需要 matplotlib）
