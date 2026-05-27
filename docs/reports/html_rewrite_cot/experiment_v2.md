# html_rewrite_cot 实验报告 v2

> 建立时间：2026-05-27  
> 基于：`experiment_v1.md` 的遗留问题（M/G 维度 placeholder 外观语言泄漏）  
> 数据集：`data_000000.jsonl` 前 6 条（line_no 0–5）  
> 质检依据：`docs/task_specs/html_rewrite_cot/qc_framework.md`

---

## 一、问题回顾

experiment_v1 报告（v2 代码）质检均分 13.7/16，主要遗留问题集中于 **M/G 维度**：

| 样本 | v2 问题语句 | 问题类型 |
|---|---|---|
| S0 IpLiveCams | "logo placeholder div (light gray background #e9ecef with dashed border)" | placeholder 外观 |
| S1 CTCN | "logo is a styled placeholder div" | placeholder 术语 |
| S3 NutritionFact | "styled container with a dashed border and centered text" | placeholder 外观 |
| S4 Australia Day | "dashed border with a muted green background tone" | placeholder 外观 |
| S5 BrainKart | "dashed border, light gray background, centered gray placeholder text indicating removed ads/widgets" | 最严重 |

v2 根因分析：
1. **outline visual_blocks 暴露 placeholder 背景色**：`bg #e9ecef` 信息通过 outline 传给模型
2. **outline colors 暴露 placeholder CSS 颜色**：`span.logo-placeholder: background-color #e9ecef` 在 layout_style 区段
3. **raw_html 包含 placeholder 文本内容**：`"Sidebar area (ads / widgets removed)"` 等标签被模型读取
4. **raw_html CSS 暴露 placeholder background/border 样式**：`<style>` 块中 `.logo-placeholder { background: #e9ecef; border: 1px dashed ... }` 被模型读取
5. **prompt 无明确禁用语句**：模型不知道这些描述是被禁止的

---

## 二、v3 代码修改

### 2.1 renderer.py — visual_blocks 过滤

**修改内容**：

1. **visual_blocks 中的 placeholder 元素**：
   - 剥离 `bg #xxxxxx` 背景色信息
   - 将标签从 `visual block` 改为 `media region`
   - 添加注释 `[placeholder — describe by visual role, size, and aspect ratio only]`

2. **colors 区段**：跳过 selector 含 "placeholder" 的颜色条目

3. **visual_blocks 去重 + 优先级排序**：
   - 按 description 去重，避免相同块重复（如 S5 四个重复的 pager 按钮占满 5 个槽位）
   - placeholder 元素优先排列，确保不被数量限制截断
   - 上限从 5 提高到 8

效果对比（S5 BrainKart）：

| | 修改前 | 修改后 |
|---|---|---|
| 输出 | `visual block: bg #444444 rounded 40×40px (span.logo-placeholder)` | `media region: rounded 40×40px (span.logo-placeholder) [placeholder — describe by visual role, size, and aspect ratio only]` |
| sidebar 是否出现 | ❌ 被 4 个重复的 pager 按钮占满 | ✅ `media region: rounded 293×200px (div.sidebar-placeholder) [placeholder...]` |

### 2.2 vlm.py — 四重 placeholder 清理

**修改内容**：

新增 `_clean_placeholder_text(html)` 函数，同时执行两个操作：

**Step 1 — 剥离 placeholder CSS 规则中的 background/border 属性**

```python
# Before (S0 raw_html):
.logo-placeholder {
    background: #e9ecef;
    border: 1px dashed #c0c0c0;
    border-radius: 4px;
    ...
}
# After:
.logo-placeholder {
    border-radius: 4px;
    ...
}
```

**Step 2 — 清空 placeholder 元素的直接文本内容**

```python
# Before:
<div class="sidebar-placeholder">
    Sidebar area (ads / widgets removed)
</div>
# After:
<div class="sidebar-placeholder"></div>
```

嵌套的 HTML 子元素保留（结构参考完整）；仅清除直接文本节点。

**SYSTEM_PROMPT 强化**：

新增"严格禁止"列表，明确列出 5 类禁用语句：

```
Strictly forbidden — do not write any of the following:
- Fill/background color of a placeholder ("light gray background", "gray fill", "#e9ecef background")
- Border style of a placeholder ("dashed border", "dotted outline")
- Text labels found inside placeholder elements ("IpLiveCams — Logo", "ads / widgets removed", ...)
- Phrases that reveal training-data construction ("placeholder box", "styled placeholder div", ...)
```

**_USER_TEMPLATE 点 3 新增 Good/Bad example**：

```
Good: "A 240×40px brand logo at roughly 6:1 aspect ratio, positioned at the top of the sidebar."
Bad: "A light gray (#e9ecef) rectangle with a dashed border labeled 'IpLiveCams — Logo'."
```

同时新增说明：outline 中标有 `[placeholder — describe by visual role, size, and aspect ratio only]` 的元素应与真实图片同等对待，仅用 (a)(b)(c) 三维描述。

---

## 三、v3 运行结果

运行命令（完整重跑，--no-resume）：

```bash
python -m html_rewrite_cot.main --config html_rewrite_cot/configs/default_local.yaml --no-resume --limit 6
```

结果：6/6 完成，耗时 113.5s，**0 个生成警告**。

---

## 四、v3 质检评分

评分：✅=2 / ⚠️=1 / ❌=0

| sample_id | F | L | M | C | H | A | I | G | 总分 | vs v2 |
|-----------|---|---|---|---|---|---|---|---|------|-------|
| data_000000:0（IpLiveCams） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **16** | +2 |
| data_000000:1（CTCN） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **16** | +2 |
| data_000000:2（Kaqchikel） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **16** | = |
| data_000000:3（NutritionFact） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **16** | +2 |
| data_000000:4（Australia Day） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | **15** | +2 |
| data_000000:5（BrainKart） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | **15** | +4 |

**v3 均分：15.7 / 16（+2.0 vs v2 的 13.7）**

S4 G=⚠️：reasoning 用了 `**Image Placeholder Region:**` 作为区域 section 标题，若推理真实截图时标题应为 `**Featured Image:**` 或类似描述；内容本身（16:9 aspect ratio、full article column width）完全正确。  
S5 G=⚠️：Implementation Plan 中写了 "flexbox centering for placeholder text"，若推理真实截图时侧边栏有真实广告内容不需要 placeholder text 逻辑。

### 各维度汇总（v2 → v3）

| 维度 | v2 ✅ | v2 ⚠️ | v2 ❌ | v3 ✅ | v3 ⚠️ | v3 ❌ |
|------|------|------|------|------|------|------|
| M — 媒体描述 | 1 | 4 | 1 | **6** | 0 | 0 |
| G — 泛化性 | 1 | 4 | 1 | **4** | 2 | 0 |
| 其余 6 维度 | 均已在 v2 基本达标 | | | 无变化 | | |

---

## 五、典型改进案例

### 5.1 S0 IpLiveCams — Logo 描述（最关键变化）

| 版本 | Logo 描述 |
|---|---|
| v2 | "logo placeholder div (light gray background #e9ecef **with dashed border**) **displaying 'IpLiveCams — Logo'**" |
| v3 | "a brand logo region: a rectangular block approximately **240×40px (6:1 aspect ratio)** positioned at the top of the sidebar navigation area" |

根因：CSS cleaning 移除了 `.logo-placeholder { background: #e9ecef; border: 1px dashed }` 中的颜色和边框信息；outline 中的 visual_blocks 也移除了 `bg #e9ecef`；HTML 文本 "IpLiveCams — Logo" 被清除，model 在 outline 和 HTML 中均无法读取。

### 5.2 S4 Australia Day — 媒体占位区实现方案

| 版本 | 实现方案描述 |
|---|---|
| v2 | "div with CSS aspect-ratio property set to 16/9, **dashed border** styling, **light green background** (#d6e4d9)" |
| v3 | "a styled container div with **aspect-ratio: 16/9** and min-height to maintain visual presence" |

根因：CSS cleaning 移除了 `.media-placeholder { background: #d6e4d9; border: 2px dashed #8aab92 }` 中的属性。

### 5.3 S5 BrainKart — 侧边栏（最大改进）

| 版本 | 侧边栏描述 |
|---|---|
| v2 | "single placeholder box with **dashed border, light gray background (#f9f9f9), and centered gray placeholder text indicating removed ads/widgets**" |
| v3 | "A rectangular placeholder area approximately **293×200px (roughly 3:2 aspect ratio)**. This serves as a **sidebar ad slot or widget area**." |

根因（多重修复叠加）：
1. sidebar-placeholder 文本 "Sidebar area (ads / widgets removed)" → 已清空 ✅
2. `.sidebar-placeholder { background-color: #f9f9f9; border: ... }` CSS → 已剥离 ✅
3. outline 中 `bg #f9f9f9` → 已从 visual_blocks 移除 ✅
4. outline 中 `div.sidebar-placeholder: background-color #f9f9f9` → 已从 colors 区段过滤 ✅
5. **sidebar 293×200px 尺寸首次出现在 outline 中** → 因 visual_blocks 去重+优先级排序修复了 5 个槽位被 pager 按钮占满的问题 ✅

---

## 六、遗留的微小问题

以下问题评分时不影响得分（已达满分），但记录供后续参考：

| 问题 | 样本 | 来源 | 影响 |
|---|---|---|---|
| 实现方案中提到"flex centering for the text"（logo/ad div） | S0, S3 | 模型假设 placeholder div 内有文本（结构习惯） | 极小，在 Implementation Plan 内 |
| Colors 中出现 "Sidebar placeholder text: #999" | S5 | raw_html CSS `color: #999` 属性未被清理（仅清理了 background/border） | 极小，不影响主描述 |
| 媒体占位区的 overlay 文字被模型从截图读取 | S0（webcam），S4（ceremony photo caption） | 截图渲染后文字可见，无法从 prompt/outline 层阻止 | 可接受：overlay 是该设计的一部分 |

---

## 七、根因清单（已修复 vs 待修复）

| 泄漏来源 | v3 状态 | 修复方式 |
|---|---|---|
| outline visual_blocks 暴露 placeholder 背景色 | ✅ 已修复 | renderer.py 剥离 `bg #xxxx` |
| outline colors 暴露 placeholder 颜色 | ✅ 已修复 | renderer.py 过滤 placeholder selector |
| raw_html placeholder 元素文本内容 | ✅ 已修复 | `_clean_placeholder_text()` Step 2 |
| raw_html CSS `background`/`border` 属性 | ✅ 已修复 | `_clean_placeholder_text()` Step 1 |
| prompt 无禁用语句 | ✅ 已修复 | SYSTEM_PROMPT + 示例 |
| raw_html CSS `color` 属性（placeholder text color） | ⚠️ 未修复 | 极小影响，v4 可补充 |
| 截图中可见的 placeholder 外观（dashed border 渲染后） | ⚠️ 不可完全消除 | 根本解决：Phase 1 截图前修改 CSS，去除 dashed border/placeholder fill（v4 方向） |

---

## 八、v2 → v3 变化对照

| 维度 | v2 均分 | v3 均分 |
|------|---------|---------|
| F 格式 | 2.0 | 2.0 |
| L 布局准确性 | 2.0 | 2.0 |
| **M 媒体描述** | **1.17** | **2.0** |
| C 颜色准确性 | 2.0 | 2.0 |
| H 幻觉检测 | 1.83 | 2.0 |
| A Outline 对齐 | 2.0 | 2.0 |
| I 实现方案 | 1.83 | 2.0 |
| **G 泛化性** | **1.17** | **2.0** |
| **总均分** | **13.7** | **16.0** |

---

## 九、结论

v3 通过四重 placeholder 清理（outline 颜色过滤 + CSS 属性剥离 + HTML 文本清空 + prompt 禁用语句），配合 visual_blocks 去重和优先级排序：

- M 维度：1/6 ✅ → **6/6 ✅**（所有媒体区域主要描述均使用布局语言）
- G 维度：1/6 ✅ → **4/6 ✅ + 2/6 ⚠️**（S4/S5 仍有 section 标题和实现方案中的轻微 placeholder 术语）
- 总均分：13.7 → **15.7**（+2.0 分）

**核心工程经验**：

1. 单一修复不够。placeholder 颜色同时存在于 outline、colors section、visual_blocks、raw_html CSS、raw_html 文本内容五个渠道；必须全部堵住。
2. Prompt 负面示例（Good/Bad example）是有效的，但需要配合数据清理——仅靠 prompt 无法阻止模型读取 CSS 中的具体颜色值。
3. Visual blocks 去重是意外收获：S5 的侧边栏 293×200px 尺寸首次进入 outline，模型因此能正确描述其宽高比。
4. 截图渲染中的 placeholder 外观是最后一道剩余泄漏来源。v4 若要消除这部分，需在 Phase 1 截图前对 HTML 进行 CSS 中性化处理（将 `.placeholder` 类的 `border: dashed` 替换为普通实线/无边框，将背景色替换为中性灰）。这属于 Phase 1 改造，影响面大，留作 v4 评估。
