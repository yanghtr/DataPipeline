# HTML 渲染管线设计文档

## 目录

1. [整体架构](#1-整体架构)
2. [日志格式与解读](#2-日志格式与解读)
3. [渲染阶段详解](#3-渲染阶段详解)
   - 3.1 CDN 离线缓存与字体鲁棒性
   - 3.2 图表动画禁用
   - 3.3 页面导航策略
   - 3.4 网络等待（切片等待 + 多轮重试）
   - 3.5 无限拉长检测
   - 3.6 文字叠压检测
   - 3.7 高宽比检测
   - 3.8 系统负载自适应
   - 3.9 浏览器管理与定期回收
   - 3.10 断点续跑与幂等输出
4. [解析阶段详解](#4-解析阶段详解)
5. [过滤阶段详解](#5-过滤阶段详解)
   - 5.1 source_file stem 的计算
   - 5.2 过滤模式
   - 5.3 统计与图表输出
   - 5.4 Resume 与统计正确性
6. [已知局限与权衡](#6-已知局限与权衡)

---

## 1. 整体架构

### 1.1 四阶段流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 1: render_html.py                                               │
│                                                                     │
│  JSON/JSONL 文件                                                     │
│      │                                                              │
│      ├─[预处理] 扫描 html 字段中的外链 URL，预下载到 CDN 缓存         │
│      │                                                              │
│      └─[并行渲染] multiprocessing.Pool × N workers                  │
│           每个 worker: sync_playwright → Chromium                   │
│                │                                                    │
│                ├─ CDN 拦截器（route）                                │
│                ├─ 图表动画禁用注入                                    │
│                ├─ goto(data:text/html;base64,...)                   │
│                ├─ 全页滚动触发 lazy paint                            │
│                ├─ 切片等待 networkidle + 无限拉长检测                 │
│                ├─ 文字叠压检测                                        │
│                ├─ 高宽比检测（tall_ratio_threshold）                  │
│                └─ full_page screenshot → {id}.png                  │
│                                                                     │
│  输出: PNG 文件 + 结构化日志(stderr，追加模式)                         │
└─────────────────────────────────────────────────────────────────────┘
                              │ stderr 追加到 render.log
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 2: parse_render_log.py                                          │
│                                                                     │
│  render.log → 正则解析 → 按 id 归组 → issues.json                   │
│                                                                     │
│  issues.json 结构:                                                   │
│    summary: 各类问题计数                                              │
│    ids: {log_id → {has_error, has_warn, categories, entries}}       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 3: filter_by_issues.py                                          │
│                                                                     │
│  issues.json + 原数据 → 剔除问题 id → 干净数据副本                    │
│  原文件不变，保留原格式（编码/BOM/缩进/分隔符）                         │
│  --summary_json 写出机器可读统计 → 供跨子目录聚合                     │
│  --stats_dir 生成 drop_ratio.png 过滤比例图                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 4: run.sh 内嵌统计                                               │
│                                                                     │
│  汇总各子目录的 summary JSON → 全局统计打印                            │
│  扫描 images_dir/*.png (PIL) → 宽度/高度/高宽比分布图 (matplotlib)    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 关键设计原则

- **幂等**：截图已存在则跳过，支持随时中断重启。
- **日志追加**：render 日志以追加模式写入，resume 后 parse+filter 看到完整历史，统计不丢失。
- **日志与渲染分离**：所有质量问题只记录日志，不影响截图输出。下游根据日志决定是否丢弃。
- **症状检测而非规则枚举**：不去识别"哪种代码会触发问题"，而是直接观测渲染后的症状（页面高度、元素位置），对未来新的触发来源自然有效。
- **可观测性**：每条日志都携带结构化 key=value 字段，便于离线统计和正则解析。
- **字体 CDN 不影响过滤**：Google Fonts 等装饰性资源失败降为 WARN，不触发记录剔除。

---

## 2. 日志格式与解读

### 2.1 日志行格式

```
[LEVEL] | [CATEGORY] | id=<log_id> | key1=val1 | key2=val2 | msg=<描述>
```

- **LEVEL**：`WARN`（问题但不致命）或 `ERROR`（渲染明确失败）。
- **CATEGORY**：问题类型，见下表。
- **log_id**：格式为 `<output_stem>:<sanitized_item_id>`，`output_stem` 由输入文件相对路径派生（见 §5.1）。

### 2.2 所有日志类别

| 类别 | 级别 | 含义 | 关键字段 |
|------|------|------|----------|
| `INFINITE_GROWTH` | WARN | 页面高度持续增长，截图可能极长 | `window_ms`, `delta_px`, `start_h`, `end_h`, `samples` |
| `TEXT_OVERLAP` | WARN | 非父子关系文字元素相互叠压 | `pairs`, `worst_ratio`, `worst_a`, `worst_b` |
| `TIMEOUT` | ERROR | goto 或 screenshot 超时 | `stage` (goto/screenshot) |
| `WAIT_TIMEOUT` | WARN | networkidle 单轮等待超时，已重试 | `attempt`, `max_wait`, `factor` |
| `WAIT_FINAL_TIMEOUT` | WARN | 两轮等待均超时，强制截图 | — |
| `REQUEST_FAILED` | ERROR/WARN | 网络请求失败（字体/装饰性 CDN 为 WARN） | `type`, `url`, `reason` |
| `HTTP_ERROR` | ERROR/WARN | HTTP 4xx/5xx 响应（字体/装饰性 CDN 为 WARN） | `type`, `status`, `url` |
| `TALL_PAGE` | WARN | 截图高宽比超过 tall_ratio_threshold | `ratio`, `width`, `height` |
| `PAGE_ERROR` | ERROR | 页面 JS 运行时抛出的异常 | `msg` |
| `CONSOLE_ERROR` | WARN | 页面 console.error 输出 | `msg` |
| `EMPTY_OUTPUT` | ERROR | 截图文件缺失或 < 1KB | `size`, `path` |
| `BAD_DATA` | ERROR | html 字段为空或纯空白 | — |
| `RENDER_ERROR` | ERROR | 其他渲染异常 | `stage`, `msg` |
| `PENDING_REQUEST` | WARN | goto 超时时仍有挂起请求 | `type`, `url` |

### 2.3 示例日志解读

**示例 1：无限拉长**
```
[WARN] | [INFINITE_GROWTH] | id=part01_output:item_123 |
    window_ms=3200 | delta_px=12000 | start_h=800 | end_h=12800 |
    samples=16 | msg=page height grew monotonically -> likely infinite stretch
```
- 在 3200ms 的观测窗口内，页面高度从 800px 增长到 12800px（净增 12000px）。
- 截图仍然产生，但会是一张 12800px 高的图，通常是废图。

**示例 2：文字叠压**
```
[WARN] | [TEXT_OVERLAP] | id=part01_output:part2026-03-23-00000_00000100 |
    pairs=2 | worst_ratio=100 | worst_a=SPAN:(Iowa State University) |
    worst_b=SPAN:(Iowa State University) | msg=overlapping text elements detected
```
- 找到 2 对叠压的文字元素对。
- 最严重的一对：两个 `<span>` 都含文字 "Iowa State University"，`worst_ratio=100` 表示重叠面积 / 较小元素面积 = 100%，即两个元素完全叠在一起。
- **典型原因**：CSS `position: absolute` 的两个同名元素被渲染到同一位置（如一个显示层覆盖在另一个上），或 CSS 动画定格在初始位置（两帧重叠）。
- 截图仍然产生。视觉上文字只显示一份，但实际上是两层叠加。

**示例 3：networkidle 超时后强制截图**
```
[WARN] | [WAIT_TIMEOUT] | id=... | attempt=0 | max_wait=6000 | factor=1.0 | msg=networkidle timeout
[WARN] | [WAIT_TIMEOUT] | id=... | attempt=1 | max_wait=9000 | factor=1.0 | msg=networkidle timeout
[WARN] | [WAIT_FINAL_TIMEOUT] | id=... | msg=proceeding to screenshot without stable signal
```
- 第一轮等待 6000ms 未达到 networkidle，第二轮放大 1.5× 等待 9000ms 也未达到，最终强制截图。
- 常见原因：页面有持续轮询的 XHR 请求（如广告、埋点），永远不会 networkidle。截图质量通常可接受。

---

## 3. 渲染阶段详解

### 3.1 CDN 离线缓存与字体鲁棒性

**背景**：大规模数据集中的 HTML 通常引用 jsDelivr / unpkg / cdnjs 等 CDN 上的 JS/CSS 库。逐条渲染时，每次都去拉网络有三个问题：
1. 速度慢（几十 ms/次 × 几十万条 = 几小时）
2. 不稳定（CDN 偶发限流或超时）
3. 不可复现（CDN 内容可能随时间变化）

**方案**：在渲染前做一次集中预下载，后续渲染时用 Playwright 的 `page.route()` 拦截所有网络请求，命中缓存就直接 fulfill，不走真实网络。

**实现细节**：

```
预下载阶段:
  _extract_urls_from_html() - 三组正则扫描 HTML:
    <script src="...">
    <link href="...">
    @import url("...")
  采样前 100 条记录（足以覆盖常见库集合）
  ThreadPoolExecutor 并发下载，原子写入（先写 .tmp 再 os.replace）

缓存键: sha256(url) + 原始扩展名
元数据: <hash>.meta 存 Content-Type（因为服务器返回的 MIME 比文件扩展名更可靠）

渲染时拦截逻辑（三级查找）:
  Level 1: 进程内 LRU 内存缓存（_BodyCache, 64条/64MB）
           → 命中则无磁盘 IO
  Level 2: 磁盘缓存目录
           → 命中则读盘 + 写入 L1
  Level 3: 真实网络
           → route.continue_()

fulfill 时强制设置:
  Access-Control-Allow-Origin: *   ← 避免 CORS 阻塞跨域字体/JS
  Cache-Control: public, max-age=31536000
```

**为什么 LRU 用进程内而不是跨进程共享**：multiprocessing.Pool 的 worker 是独立进程，共享内存实现复杂、有锁竞争。同一个进程渲染的 N 条记录大概率使用相同的库，进程内 LRU 已能大幅减少磁盘读。

**字体 CDN 不可达的处理**：Google Fonts、fonts.gstatic.com 等在某些网络环境下频繁返回 404 或超时（CDN 预下载失败），但字体缺失只影响视觉样式，不影响页面结构和核心内容渲染。因此：
- `REQUEST_FAILED` / `HTTP_ERROR`：若资源类型为 `font`，或 URL 域名属于 `_DECORATIVE_DOMAINS`（Google Fonts、FontAwesome 等），降级为 **WARN**，不产生 ERROR。
- 这样字体失败不会导致记录被过滤，但仍可通过 `--level warn` 或手动查看日志追踪。

### 3.2 图表动画禁用

**背景**：`page.screenshot(animations="disabled")` 只能压制 CSS 动画（transitions/keyframes）。JS 驱动的 canvas 动画（Chart.js、ECharts 的入场动画）不受影响：截图时图表可能还在"绘制过程中"，只画了一半。

**方案**：在页面加载时，通过 `Object.defineProperty` 拦截 `window.Chart` 和 `window.echarts` 的赋值时机，在库刚加载时就修改其默认配置：

```javascript
// Chart.js: 关闭所有动画
Chart.defaults.animation = false;
Chart.defaults.animations = false;
Object.keys(Chart.defaults.transitions).forEach(k => {
    Chart.defaults.transitions[k] = { animation: { duration: 0 } };
});

// ECharts: 包装 init，使每个实例的 setOption 自动注入 animation: false
const origInit = echarts.init;
echarts.init = function(...args) {
    const inst = origInit.apply(this, args);
    const origSetOption = inst.setOption;
    inst.setOption = function(opt, ...rest) {
        opt.animation = false;
        return origSetOption.call(this, opt, ...rest);
    };
    return inst;
};
```

**为什么用 `Object.defineProperty` 而不是直接赋值**：CDN 资源是异步加载的，`window.Chart` 在 `add_init_script` 执行时还不存在。用 property setter 可以在库被赋值到 `window` 的瞬间插入修改逻辑，时机精确。

**通过 `add_init_script` 注入**：Playwright 的 `add_init_script` 在每个新页面的任何脚本执行前注入，包括比 DOMContentLoaded 更早，确保即使是内联 `<script>` 中的 `new Chart(...)` 也能被拦截。

### 3.3 页面导航策略

**为什么用 `data:text/html;base64,...` 而不是写临时文件再 `file://`**：
- 无需管理临时文件的创建/清理
- 无文件系统权限问题
- base64 编码后的 URL 在 Playwright 中完全受控，不会触发额外的文件协议行为

**`wait_until="domcontentloaded"` 而不是 `"load"` 或 `"networkidle"`**：
- `"load"` 要等所有资源（包括图片）加载完，CDN 资源未缓存时会很慢
- `"networkidle"` 作为 goto 的条件太严格，容易超时
- `domcontentloaded` 只等 HTML 解析完，后续资源加载由专门的等待循环处理

**全页滚动**：goto 完成后立即执行一次从顶到底再回顶的滚动（每 300px 步进，每步 50ms 延迟）。目的是触发：
- `IntersectionObserver` 回调（懒加载内容）
- `visibility` 相关的 CSS 动画
- 滚动触发的数据请求

### 3.4 网络等待（切片等待 + 多轮重试）

**原始方案的问题**：
```python
page.wait_for_load_state("networkidle", timeout=max_wait_ms)  # 阻塞式，无法插入检测
```
`max_wait_ms` 期间整个进程阻塞，无法做周期性的无限拉长检测。如果页面永远不 networkidle，只能等满超时才能截图。

**改进方案：切片等待**：
```
每轮:
  elapsed = 0
  while elapsed < cur_max_wait:
    step = min(500ms, 剩余)
    try:
      wait_for_load_state("networkidle", timeout=step)
      → 命中: wait_ok=True, 跳出
    except Timeout:
      pass
    elapsed += step
    插入 _check_growth_and_warn()  ← 关键：命中拉长则提前终止
  
  如果未 wait_ok 且未命中拉长: 第二轮 cur_max_wait *= 1.5
```

**两轮重试的意义**：部分页面在第一轮等待结束时仍有少量请求未完成，稍等片刻就能稳定。1.5× 放大系数对应"如果页面比预期慢，多给一些时间"的直觉。

**为什么切片是 500ms**：太小（如 50ms）导致 wait_for_load_state 频繁抛出超时异常，有额外开销；太大（如 2000ms）使拉长检测的响应延迟过高。500ms 是对"检测实时性"和"API 调用开销"的折中。

### 3.5 无限拉长检测

**触发场景**：
部分 HTML 存在设计缺陷，会让页面 `scrollHeight` 随时间持续增长，最终截到一张几万像素高的废图。常见原因包括：
- Chart.js 配置 `maintainAspectRatio: false` + 父容器无固定高度 → Canvas 试图填满父容器 → 父容器因 Canvas 变高而变高 → 循环
- ECharts 同类问题
- `ResizeObserver` 回调触发 DOM 变化 → 触发 resize → 死循环
- 错误的 CSS（`height: 100%` 链式传递到根节点）
- `setInterval` 持续 `appendChild`

**为什么用高度采样而不是静态 HTML 分析**：
静态分析需要枚举每种触发模式（maintainAspectRatio、height 链式、ResizeObserver...），永远无法穷举新出现的模式。"页面高度一直在涨"是所有触发来源的共同症状，直接观测症状更鲁棒。

**算法详解**：

```
注入的 JS 采样器:
  每 200ms 采样一次 max(scrollHeight_html, scrollHeight_body)
  保留最近 600 个样本（覆盖约 2 分钟，足够所有渲染场景）

_analyze_growth() 判定逻辑（全部条件需同时满足）:
  1. 窗口时长 >= growth_min_window_ms (默认 1000ms)
     → 排除"页面刚加载时的正常布局调整"（通常在前几百毫秒内完成）
  
  2. 窗口内高度单调不减（允许 ±2px 的抖动容忍）
     → 排除"先涨后缩"的合法内容（如折叠面板展开后收起）
  
  3. 净增长 >= growth_threshold_px (默认 500px)
     → 排除"高度确实在增长但量很小"的正常加载（如字体加载导致的微小回流）
  
  4. 末尾样本仍在增长（window[-1].h > window[-2].h）
     → 排除"先涨后稳"：页面渲染初期高度增长后稳定是正常的
     → 只有"到截图前一刻还在涨"才判定为异常
```

**为什么截图前还要"补等"**：
当 CDN 全部缓存命中时，`networkidle` 几乎瞬间触发（<100ms），切片等待循环来不及积累满 `growth_min_window_ms` 的样本就结束了，导致漏判。截图前检查样本窗口时长是否足够，不足则补等。

**命中后仍然截图的设计意图**：
已有 `WARN` 日志，下游可以：
- 丢弃（依据日志 filter）
- 保留（部分"拉长"图可能仍有用）
- 按高度阈值决定（如 > 5000px 丢弃）
渲染器不做此决策，保持单一职责。

### 3.6 文字叠压检测

**触发场景**：
- `position: absolute/fixed` 的元素叠放在同位置（常见于 JS 动态定位的 tooltip/label）
- 多个元素的 CSS 变换导致的意外重叠
- HTML 模板错误（同一内容被渲染两次）

**算法**：

```
JS 注入在截图前执行:

1. 收集候选元素（≤ 300 个）:
   标签: p, h1-h6, span, li, td, th, label, a, button
   过滤条件（任一满足则跳过）:
     - 无直接文本子节点（childNodes 中无非空 TEXT_NODE）
     - display:none / visibility:hidden / opacity < 0.1
     - 有 transform 祖先（getBoundingClientRect 被变换坐标污染，不可靠）
     - 宽度或高度 < 5px

2. 两两比较 BoundingClientRect（O(N²)，N≤300，上限 20 对）:
   排除条件（任一满足则不算叠压）:
     a. 祖先-后代关系（el.contains() 检测）
        → <h1><span>文字</span></h1> 的边界框天然重叠，合法
     b. 同 parent 且都在 normal flow（position: static 或 relative）
        → 块级/行内元素的正常排列，视觉上紧挨但 rect 可能有像素级交叉，属误差
   判定阈值:
     重叠宽度 > 5px AND 重叠高度 > 5px
     AND 重叠面积 / min(元素A面积, 元素B面积) > threshold（默认 0.20）

3. 记录最多 20 对，WARN 日志含 worst_ratio 和元素文本片段
```

**为什么是 `min(面积A, 面积B)` 而不是 `面积A + 面积B`**：
用较小元素的面积做分母，描述的是"较小元素有多少比例被覆盖"。如果用两元素面积之和，一个大容器和一个小标签轻微重叠可能显示很低的比例，但视觉上标签其实被完全遮住了。

**`worst_ratio=100` 的含义**：
重叠面积 / min(面积A, 面积B) = 1.0（100%），即两元素完全重叠，通常是两个绝对定位的相同元素叠在同一坐标。

### 3.7 高宽比检测

截图完成后，用 PIL 读取图片尺寸，若 `height / width > tall_ratio_threshold`（默认 4.0）则记录 `WARN TALL_PAGE`。

**与无限拉长检测的关系**：
- 无限拉长检测在渲染过程中（等待 networkidle 期间）运行，命中时截图仍然产生。
- 高宽比检测在截图后运行，作为补充保险：即使拉长检测漏判（如页面慢速增长但超过阈值才停），高宽比检测仍能捕获异常长图。
- 两者都只记录 WARN，由下游过滤策略决定是否丢弃。

`tall_ratio_threshold` 在 `run.sh` 中默认 **4.0**，可通过 `--tall_ratio F` 调整。

### 3.8 系统负载自适应

**背景**：在多任务共享的机器上（如集群节点），系统负载高时，Playwright 的 `wait_for_load_state` 和截图都更慢。如果超时参数是固定的，高负载时误判率显著上升。

**方案**：读取 `os.getloadavg()` 的 1 分钟平均负载，按 CPU 数量归一化：

| 负载/CPU | 倍率 | 说明 |
|----------|------|------|
| < 0.8 | 1.0× | 轻载，不调整 |
| 0.8 ~ 1.5 | 2.0× | 中等负载 |
| 1.5 ~ 2.5 | 3.5× | 高负载 |
| > 2.5 | 5.0× | 严重过载 |

`base_max_wait` 和 `base_idle_timeout` 均乘以此倍率。

**Windows 上无 `getloadavg`**：退化为 1.0×，不影响功能。

### 3.9 浏览器管理与定期回收

**架构**：主进程 → `multiprocessing.Pool(N)` → N 个 worker 进程，每个 worker 独立持有一个 Chromium 实例。

**为什么用进程而不是线程**：Playwright 的 sync API 是 blocking I/O，多线程会因 GIL 和 Playwright 内部事件循环的交互导致问题。多进程可以真正并行，且崩溃隔离好。

**浏览器定期回收**（`recycle_every`，默认 200 张）：
Chromium 长时间运行会出现内存碎片积累、句柄泄漏、偶发的渲染器崩溃。每渲染 N 张后主动 `close()` 浏览器再重新 `launch()`，保持进程健康。代价是每次重启的冷启动时间（~1s），但相比积累的问题可接受。

**进度条聚合**：N 个 worker 进程通过 `multiprocessing.Manager().Queue()` 汇报进度，主进程的 `threading.Thread` 消费队列更新 tqdm。

### 3.10 断点续跑与幂等输出

**断点续跑**：每次运行前 `os.scandir(out_dir)` 列出已存在的 `*.png`，构建 `existing` 集合，跳过已完成的 id。即使中途中断，重启后从断点继续，不重复渲染。

**日志追加**：run.sh 使用 `>>` 追加模式写 render.log，保留历史 WARN/ERROR。后续 parse 解析完整累积日志，filter 基于完整 issues.json 过滤，统计不因 resume 而失真。使用 `--no-resume` 可清空日志重新记录（已有 PNG 不受影响）。

**统计正确性**：
- 若 render 日志以 `>` 覆盖，resume 跳过的记录不在新日志里，parse 只能看到"这次渲染的"问题，filter 会漏掉历史问题记录 → 过滤不充分。
- 追加模式下，parse 始终看到全量历史，filter 统计准确。

**幂等 ID 处理**：
- 原始 id 字段（如 URL、含冒号的 record_uid）不能直接用作文件名。
- `_sanitize_id()` 将非安全字符替换为 `_`，截断到 200 字符。
- 相同的原始 id 始终产生相同的文件名，保证幂等。

**失败时写白图**：渲染异常时，如果输出文件不存在，写一张 1280×960 的白色 PNG 占位。下次扫描时这个 id 会被标记为"已存在"而跳过，避免反复重试一个永远失败的 id（如 html 字段为空）。真正需要重试时，手动删除对应的白图即可。

**`EMPTY_OUTPUT` 检测**：截图后检查文件大小，< 1KB 记 ERROR。正常截图通常 >50KB，<1KB 说明截图了一个空白或几乎空白的页面。

---

## 4. 解析阶段详解

### 4.1 日志行解析

```
正则: ^\[(WARN|ERROR)\]\s*\|\s*\[([^\]]+)\]\s*\|\s*id=([^\s|]+)(.*)?$
```

- 第1组：LEVEL
- 第2组：CATEGORY
- 第3组：log_id（形如 `stem:sanitized_id`）
- 第4组：剩余字段（`key=value | key=value | msg=...`）

`log_id` 按最后一个 `:` 分割为 `source_file` 和 `item_id`。

### 4.2 issues.json 结构

```json
{
  "summary": {
    "total_flagged_ids": 42,
    "total_entries": 87,
    "by_level": {"WARN": 60, "ERROR": 27},
    "by_category": {"TEXT_OVERLAP": 35, "INFINITE_GROWTH": 12, "TIMEOUT": 8, ...},
    "warn_ids_count": 38,
    "error_ids_count": 15
  },
  "ids": {
    "stem:item_id": {
      "item_id": "item_id",
      "source_file": "stem",
      "has_error": false,
      "has_warn": true,
      "categories": ["TEXT_OVERLAP"],
      "entries": [
        {
          "level": "WARN",
          "category": "TEXT_OVERLAP",
          "fields": {"pairs": 2, "worst_ratio": 100, ...},
          "log_source": "render.log",
          "lineno": 1234
        }
      ]
    }
  }
}
```

---

## 5. 过滤阶段详解

### 5.1 source_file stem 的计算

渲染时，输出目录名（stem）由输入文件相对 `json_dir` 的路径决定：

```
json_dir = /data/run/batch01
file     = /data/run/batch01/part_01/output.jsonl
rel      = part_01/output.jsonl
parts    = ["part_01", "output"]
stem     = "part_01_output"
```

过滤时必须传入相同的 `--json_dir`，脚本才能计算出相同的 stem，正确匹配 issues.json 中的 `source_file`。

### 5.2 过滤模式

| `--level` 值 | 剔除哪些 id |
|-------------|------------|
| `all`（默认） | issues.json 中出现的所有 id（WARN 和 ERROR） |
| `error` | 只剔除有 ERROR 的 id（纯 WARN 的保留） |
| `warn` | 只剔除有 WARN 的 id（纯 ERROR 的保留） |

**如何选择**：
- `error`：最保守，只去掉渲染明确失败的数据（超时、空白、JS 崩溃），保留有轻微质量问题的数据。
- `all`：最严格，同时去掉所有有告警的数据（如文字叠压、无限拉长）。
- `warn`：较少使用，适合只想去掉视觉质量问题但保留渲染失败记录的场景。

通常推荐 `error` 级别先过一遍，再根据实际需要决定是否用 `all` 进一步清洗。

### 5.3 统计与图表输出

`--summary_json FILE` 写出机器可读 JSON：
```json
{
  "original":          3638,   // 实际参与过滤的文件总条目（不含直通文件）
  "removed":           495,
  "kept":              3143,
  "passthrough_items": 3649,   // 直通文件（无对应 issue stem）的条目总数
  "panguml_written":   3143,
  "panguml_skipped":   3649
}
```

run.sh 在所有子目录过滤完成后，用内嵌 Python 聚合各子目录的 summary JSON，打印全局统计。

`--stats_dir DIR` 生成 `drop_ratio.png`（各过滤文件的剔除比例柱状图，需要 matplotlib）。阶段 4 另外生成 `image_dimensions.png` 和 `image_ratio.png`（所有 PNG 的尺寸/高宽比分布，需 PIL + matplotlib）。

**直通文件与过滤文件分离**：`api_calls.jsonl` 这类没有对应 stem 的文件归为"直通"，统计上与实际过滤的 `output.jsonl` 分开显示，避免两者混在总条目数里产生误导。

### 5.4 Resume 与统计正确性

render 日志以追加模式（`>>`）写入，每次 parse 解析完整累积日志，filter 基于累积 issues.json 做过滤：
- resume 续跑时，已渲染记录的历史 WARN/ERROR 仍在日志中 → filter 正确剔除历史问题 id
- 全新跑（`--no-resume`）时，`run.sh` 先清空日志文件，再以 `>>` 追加

### 5.5 JSONL 格式的处理

JSON 数组和 JSONL 的过滤方式不同：
- **JSON 数组**：整个文件 `json.load()` → 过滤 list → `json.dump()` 写回，自动探测原文件缩进/BOM/分隔符保持一致。
- **JSONL**：逐行读取 → 解析 → 判断 id 是否在 blocked 集合 → 保留则原样写回（保持原始行格式，包括换行符类型）。

**为什么 JSONL 不重新 dump**：重新 `json.dumps()` 每行会损失原始格式（如 key 顺序、空格风格），逐行原样写回更安全。

### 5.6 ID 匹配的一致性

过滤时用相同的 `_sanitize_id()` 处理原始 id，与渲染时写入 log 的 item_id 保持一致：

```
原始 id: "part2026-03-23-00000:00000258"
渲染时 → _sanitize_id() → "part2026-03-23-00000_00000258"  （写入 log）
过滤时 → _sanitize_id() → "part2026-03-23-00000_00000258"  （查 blocked 集合）
```

---

## 6. 已知局限与权衡

### 6.1 TEXT_OVERLAP 的误报

以下情况会被错误地报告为叠压：
- **绝对定位的装饰性文字**（如水印、角标），与正文内容有意重叠
- **CSS clip/overflow:hidden 截断**的元素，视觉上不可见但 getBoundingClientRect 仍有重叠坐标
- **极小的字体/行高差异**导致相邻元素有 1-2px 交叉（已通过 >5px 阈值过滤大部分此类情况）

`overlap_ratio_threshold` 默认 0.20，可上调到 0.5 减少误报（但会漏报轻微叠压）。

### 6.2 INFINITE_GROWTH 的漏报

以下情况会漏报：
- **拉长发生在 goto 之后但在采样器启动之前**：已通过 `add_init_script` + 补等机制尽量避免
- **拉长速度极慢**（如每秒增长 10px），在观测窗口内净增长 < `growth_threshold_px` 阈值
- **周期性涨落**（涨→缩→涨），单调性检测会排除，但最终结果可能是高度超标

### 6.3 networkidle 等待的局限

以下页面永远不会达到 networkidle：
- 广告、统计、IM 的长轮询 XHR
- WebSocket 连接
- 定时刷新的数据看板

这些页面最终会走到 `WAIT_FINAL_TIMEOUT` 后强制截图。通常截图质量可以接受（内容加载完了，只是网络请求没断），但有时动态内容未完全渲染。

`--base_max_wait` 可适当调大，但等待时间与吞吐量成反比。

### 6.4 data: URL 的长度限制

对于超大 HTML（>几 MB），base64 编码后的 data: URL 可能超过浏览器限制（Chromium 大约 256MB，实际取决于系统内存）。当前代码未做分割处理，超大 HTML 会导致 `goto` 失败并记录 `RENDER_ERROR`。
