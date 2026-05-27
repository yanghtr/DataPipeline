# html_rewrite_cot 实验报告 v1

> 建立时间：2026-05-27  
> 数据集：`data_000000.jsonl` 前 6 条（line_no 0–5）  
> 质检依据：`docs/task_specs/html_rewrite_cot/qc_framework.md`

---

## 一、实验目标

验证两阶段 CoT 数据构造流水线（Phase 1: HTML Outline 提取 / Phase 2: VLM reasoning 生成）的输出质量，重点评估：

1. Outliner 提取的结构信息是否准确、无噪声
2. VLM reasoning 的媒体区域描述是否使用**布局语言**而非 placeholder 外观语言（泛化能力核心指标）
3. 各格式/结构指标是否达标

---

## 二、第一轮运行（v1）— 发现问题

第一轮运行完成后，通过 debug JSONL 分析发现多项系统性问题：

### 2.1 Outliner 问题

| 问题 | 表现 | 影响 |
|---|---|---|
| Nav 导航链接双重出现 | `navigation_links` 和 `lists` 同时包含 `<nav>` 内的 `<ul>` | outline_text 冗余，Outline↔Reasoning 对齐度(A)下降 |
| 裸 tag 选择器出现在 layout_hints | `a: horizontal flex row`、`div: flex column` 等无 class/id 的噪声条目 | outline_text 噪声，A维度影响 |
| `h1`/`h2`/`li` 被标为 "full-width bar (likely nav/header)" | `style_hint` 触发条件未排除文本标签 | outline_text 错误信息，L/H维度影响 |
| `ol`（面包屑）被标为 "full-width bar" | `_TEXT_TAGS` 未包含 `ol` | 同上 |
| Logo placeholder div 完全未被 visual_blocks 捕获 | STRUCT_RE 无 `logo\|placeholder\|media`，isVB 文本阈值=15 过小 | Logo 不在 outline 中，A/M维度显著下降 |

### 2.2 VLM Prompt 问题

| 问题 | 表现 | 影响 |
|---|---|---|
| 媒体描述使用 placeholder 外观语言 | reasoning 中出现 "gray box", "dashed border", "overlay text" | M/G维度大量失分 |
| 媒体实现方案描述过窄 | "all image regions will be implemented as `<img>` or styled placeholder divs" | 排除了 SVG、CSS background、video 等合法实现方式 |
| 过渡句缺乏指导 | 模型有时省略过渡句或直接开始写 HTML | F维度问题 |

### 2.3 Postprocess 问题

| 问题 | 表现 | 影响 |
|---|---|---|
| `region_section_count` 对 `**Bold:**` 格式头部计数为 0 | 正则未剥离 markdown 加粗符号 | F维度误判 |
| `image` alignment warning 误报 | reasoning 总含 "analyze the image" 触发关键词 | 生成无意义警告 |
| `table` alignment warning 误报 | reasoning 常说 "avoid table layout" 触发关键词 | 同上 |

### 2.4 Phase 2 Resume 问题

SIGKILL 情况下，panguml 输出已写但 `phase2_done.txt` 未更新，导致重跑时重复处理。

---

## 三、代码修改记录

### Round 1 — Outliner 基础修复

**文件：`html_rewrite_cot/pipeline/outliner.py`**

```python
# 跳过 <nav> 内的 <ul>（已在 navigation_links 处理）
if lst.find_parent("nav"):
    continue

# 过滤裸 tag 选择器（无 class/id）
_has_qualifier = "." in sel or "#" in sel
if "flex" in display and _has_qualifier:
    layout_hints.append(...)
```

### Round 2 — 文本标签 + Postprocess 修复

**文件：`html_rewrite_cot/pipeline/outliner.py`**

```python
_TEXT_TAGS = {"h1","h2","h3","h4","h5","h6","p","li","span","a","label","strong","em"}
if w >= viewport_w * 0.85 and 0 < h < 120 and tag not in _TEXT_TAGS:
    style_hints.append(f"{sel}: full-width bar (likely nav/header)")
```

**文件：`html_rewrite_cot/pipeline/postprocess.py`**

```python
# 修复 **Bold:** 格式的 section 计数
stripped = re.sub(r"^\*+\s*|\s*\*+$", "", stripped)
if stripped.endswith(":") and 3 < len(stripped) <= 60:
    region_section_count += 1

# 移除误报的 alignment 检查项
checks = [
    ("form", ["form","input","textarea","select"]),
    ("sidebar", ["aside","sidebar","panel"]),
    ("footer", ["footer","foot"]),
    # "table" 和 "image" 已移除
]
```

### Round 3 — Logo 捕获 + Prompt 重写

**文件：`html_rewrite_cot/pipeline/outliner.py`**

```python
# ol 加入 _TEXT_TAGS（面包屑 ol 不应触发 full-width bar）
_TEXT_TAGS = {..."ol"...}
```

在 `_PLAYWRIGHT_JS` 中：

```js
// 扩展 STRUCT_RE 以捕获 logo/placeholder/media div
const STRUCT_RE = /container|wrapper|layout|grid|row|col|sidebar|hero|banner|card|panel|
    nav|header|footer|main|content|column|block|section|logo|placeholder|media/i;

// 提高 isVB 文本阈值（15 → 50），捕获含较长标签的 logo placeholder
const isVB = r.width*r.height > 400 && txt.length < 50 && (hasBg || hasBgImg || hasBorder);
```

**文件：`html_rewrite_cot/pipeline/vlm.py`**

SYSTEM_PROMPT 新增媒体区域描述指导：
- 明确说明 placeholder 外观特征（dashed border, gray fill 等）是训练数据特有，不应描述
- 强调根据实际 HTML 结构判断实现技术（`<img>` / CSS background / video / SVG / placeholder div）

`_USER_TEMPLATE` 更新：
- 媒体区域描述新增 (a)(b)(c) 三维要素：视觉角色、相对尺寸、宽高比
- 明确要求"Describe what the region IS and WHERE it sits — not how the placeholder box looks"
- Implementation Plan 中要求说明每个媒体区域的实现技术和尺寸控制方式
- 增加过渡句示例，要求用自己的措辞，不照抄示例

**文件：`html_rewrite_cot/phase2_generate.py`**

Resume 修复：扫描 debug JSONL 补充已完成 ID：

```python
if config.output.debug_dir:
    for jsonl_file in by_file:
        _, dbg_path = _make_output_paths(config, jsonl_file)
        if dbg_path and dbg_path.exists():
            with open(dbg_path) as f:
                for line in f:
                    d = json.loads(line.strip())
                    done_ids.add(d.get("sample_id"))
```

---

## 四、第二轮运行（v2）— 质检结果

运行命令：

```bash
python -m html_rewrite_cot.main --config html_rewrite_cot/configs/default_local.yaml
```

输出：6/6 条完成，耗时 101.2s，0 个生成警告。

### 4.1 质检评分表

评分：✅=2 / ⚠️=1 / ❌=0

| sample_id | F | L | M | C | H | A | I | G | 总分 | 主要问题 |
|-----------|---|---|---|---|---|---|---|---|------|---------|
| data_000000:0（IpLiveCams） | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ⚠️ | **14** | Logo 描述仍含 "dashed border, #e9ecef" 等外观语言 |
| data_000000:1（CTCN） | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ⚠️ | **14** | Impl plan 中称 logo 为 "styled placeholder div" |
| data_000000:2（Kaqchikel） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **16** | 最优样本，媒体描述完全泛化 |
| data_000000:3（NutritionFact） | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ⚠️ | **14** | Logo: "styled container with a dashed border" |
| data_000000:4（Australia Day） | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ✅ | ✅ | ⚠️ | **13** | Media placeholder 描述含 bg color 和 dashed border；H 部分细节难以从截图验证 |
| data_000000:5（BrainKart） | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ⚠️ | ❌ | **11** | 侧边栏描述完全使用 placeholder 外观 + "ads/widgets removed" 文本 |

**平均分：13.7 / 16**

### 4.2 各维度汇总

| 维度 | ✅ 通过 | ⚠️ 可接受 | ❌ 问题 | 说明 |
|------|---------|-----------|---------|------|
| F — 格式 | 6/6 | 0/6 | 0/6 | 全部 section 存在，无 HTML/fence 残留 |
| L — 布局准确性 | 6/6 | 0/6 | 0/6 | 主要区域和布局结构全部正确 |
| M — 媒体描述 | 1/6 | 4/6 | 1/6 | S2 最优；S5 侧边栏完全使用 placeholder 外观 |
| C — 颜色准确性 | 6/6 | 0/6 | 0/6 | 颜色值均可在 outline 或 raw_html 中找到对应 |
| H — 幻觉检测 | 5/6 | 1/6 | 0/6 | S4 部分细节（"5段正文"等）难以精确验证 |
| A — Outline 对齐 | 6/6 | 0/6 | 0/6 | Logo 信息修复后均在 outline 中有对应 |
| I — 实现方案 | 5/6 | 1/6 | 0/6 | S5 侧边栏无 aspect ratio 控制说明 |
| G — 泛化性 | 1/6 | 4/6 | 1/6 | 与 M 维度正相关，S5 最差 |

---

## 五、v1 → v2 关键改进

### 5.1 Outliner 改进效果

| 样本 | v1 visual_blocks 状态 | v2 visual_blocks 状态 |
|---|---|---|
| S0 IpLiveCams | 无 logo 信息 | `span.logo-placeholder: bg #e9ecef rounded 240×40px` ✅ |
| S1 CTCN | 无 logo 信息 | `div.logo-placeholder: bg #495057 rounded 160×48px` ✅ |
| S2 Kaqchikel | 部分 logo 信息 | 完整 3 个媒体区域（header 940×286, footer 180×62 + 70×70） ✅ |
| S3 NutritionFact | 无 logo 信息 | logo placeholder 现在被捕获 ✅ |
| S4 Australia Day | 无 media-placeholder | `div.media-placeholder` 出现在 style_hints ✅ |
| S5 BrainKart | 无 logo/sidebar 信息 | `span.logo-placeholder: bg #444444 40×40px` + `div.sidebar-placeholder: 293×200px` ✅ |

### 5.2 VLM reasoning 改进效果

**M / G 维度典型案例：**

S1 CTCN logo（v2）：
> "a rectangular box approximately **160×48px (roughly 3.3:1 aspect ratio)**"

S4 Australia Day 媒体区域（v2）：
> "Its visual role is as a **featured content image**, with approximately a **16:9 aspect ratio**. Implementation will use a div with CSS **aspect-ratio property set to 16/9**"

S2 Kaqchikel 页脚（v2，最优）：
> "left column contains a rectangular logo placeholder (**180 × 62 pixels, aspect ratio ~2.9:1**)"  
> "right column contains a **square badge placeholder (70 × 70 pixels, 1:1 aspect ratio)**"

**F 维度改进：** region_section_count 现在正确统计 `**Bold:**` 格式的 section 头部（修复 postprocess 正则）。

**Outliner 噪声消除：** 无更多 `a: horizontal flex row` 等裸 tag 选择器；面包屑 `ol` 不再触发 "full-width bar" 误报；无 h1/h2 被标为导航条。

---

## 六、遗留问题与下一版迭代建议

### 6.1 M/G 维度残留问题（核心）

最主要的残留问题：**模型仍然会在 logo 和 sidebar placeholder 的描述中使用 placeholder 外观语言**。

| 样本 | 问题语句 | 类型 |
|---|---|---|
| S0 | "logo placeholder div (light gray background #e9ecef with **dashed border**)" | placeholder 外观 |
| S1 | "logo is a **styled placeholder div**" (in impl plan) | placeholder 术语 |
| S3 | "styled container with a **dashed border** and centered text" | placeholder 外观 |
| S4 | "light green background (#d6e4d9) with a **dashed border** (#8aab92)" | placeholder 外观 |
| S5 | "**dashed border**, light gray background… centered gray placeholder text **indicating removed ads/widgets**" | 最严重：外观 + placeholder 文本内容 |

**根因分析：**

1. Logo 区域：outline 中的 visual_blocks 信息（如 `bg #e9ecef rounded 240×40px`）本身就包含背景色，模型倾向于直接复述这些视觉属性，而这些属性正是 placeholder 的外观。
2. Sidebar 区域（S5）：`div.sidebar-placeholder: 293×200px` 在 outline 中带有 "placeholder" 标签，且 HTML 中有 "ads/widgets have been removed" 文本，模型读取了这些信息。
3. 颜色（如 `#d6e4d9` 占位背景）出现在 outline 的 visual_blocks 中，导致模型将其纳入描述。

**修复方向（v2 → v3）：**

- **Outliner 过滤**：visual_blocks 只导出 `width×height` 和 `tag+class`，不导出背景色（或单独标注"this is a placeholder bg, do not describe in reasoning"）；或将 placeholder 的背景色不暴露给 reasoning，只保留尺寸信息。
- **Prompt 负面示例**：在 prompt 中加入明确的**禁用语句模式**（Negative examples），如"Do not write phrases like 'a div with dashed border', 'light gray background placeholder', or 'indicating removed ads/widgets'"。
- **HTML placeholder 文本屏蔽**：对 raw_html 中出现的 "ads/widgets have been removed"、"Image Placeholder"、"Logo" 等 placeholder 文本，在传给 VLM 前预处理删除或替换为通用描述。

### 6.2 其他遗留问题

| 问题 | 严重程度 | 建议 |
|---|---|---|
| S4 H 维度：部分细节（"5段正文"）难以验证 | 低 | 无需处理，属于正常估算误差范围 |
| S5 I 维度：sidebar 无 aspect-ratio 控制说明 | 低 | 提高 M/G 后自然改善 |
| `<style>` 在 reasoning 中提及触发旧版警告 | 极低 | postprocess 检测已移除此误报 |
| Phase 2 resume 仍依赖 debug_dir 扫描 | 中 | 可考虑用 SQLite 或独立 state 文件替代 |

### 6.3 下一迭代优先级

1. **最高优先**：解决 M/G 维度的 placeholder 外观语言问题（prompt 负面示例 + 可能的 HTML 预处理）
2. **中优先**：测试更大批量（50条）验证修复效果的稳定性
3. **低优先**：Phase 2 resume 机制完善

---

## 七、结论

v2 版本在 F/L/C/A 维度上达到全样本满分，整体均分 13.7（≥10 门槛，接近 ≥14 优秀线）。核心改进集中于 Outliner 的 logo 捕获能力和 VLM prompt 的媒体描述指导。

主要遗留问题集中于 M/G 维度（媒体描述泛化性），模型仍倾向于复述 placeholder 的外观颜色和边框样式。该问题需在下一个 prompt 版本或 HTML 预处理层面解决，是 v3 迭代的核心工作方向。
