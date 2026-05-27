# html_rewrite_cot CoT 数据质检框架

> 版本：v1.0  
> 建立时间：2026-05-27  
> 适用范围：html_rewrite_cot 流水线生成的 CoT 训练数据，debug JSONL 格式

---

## 一、质检维度定义

共 8 个维度，每维度评分：✅ 通过 / ⚠️ 有瑕疵但可接受 / ❌ 有问题

### F — 格式合规

检查项：
- [ ] 存在且仅存在 6 段固定结构（任务引导句 → Layout Analysis → 区域段落 → Colors Observed → Structure and Implementation Plan → 过渡句）
- [ ] 过渡句自然，不与 Implementation Plan 段落混用，独立于最后
- [ ] 无 HTML/CSS 代码内容（`contains_html_in_reasoning = false`）
- [ ] 无 markdown code fence 残留（`contains_markdown_fence_in_reasoning = false`）
- [ ] 各必须 section 存在（`has_layout_analysis_section`, `has_colors_observed_section`, `has_structure_implementation_plan_section` 均为 true）

评分依据：
- ✅：全部通过
- ⚠️：过渡句格式轻微异常（如未换行），或 region_section_count 低但确认有区域内容
- ❌：缺少固定 section，或包含 HTML/代码

---

### L — 布局准确性

检查方法：对照截图，逐项核验 reasoning 中的布局描述。

检查项：
- [ ] 列数（单列/双列/三列）正确
- [ ] 主要区域（header/main/sidebar/footer）顺序从上到下正确
- [ ] 列宽比例描述与截图视觉吻合（允许 ±10% 估算误差）
- [ ] 重要子区域（breadcrumb、pager、widget 等）均有描述
- [ ] 布局技术（flexbox/grid）判断正确

评分依据：
- ✅：主要区域和布局结构全部正确
- ⚠️：细节区域有遗漏，但核心结构正确
- ❌：主要布局结构（列数/区域顺序）错误

---

### M — 媒体区域描述质量

**这是训练数据泛化能力最关键的维度。**

核心原则：训练数据中图片为 placeholder，但描述必须用布局属性而非 placeholder 外观特征，以便模型在真实截图场景下也能使用同样的描述框架。

检查项：
- [ ] 每个媒体/图片区域均有独立描述
- [ ] 描述包含：视觉角色（hero banner / logo / thumbnail / ad / avatar 等）
- [ ] 描述包含：相对尺寸（占内容区宽度的比例，或绝对估算尺寸）
- [ ] 描述包含：宽高比（如 16:9、1:1、3.5:1 等）
- [ ] **不包含** placeholder 专有外观描述：dashed border、gray fill、overlay text label、"styled placeholder div"等表述
- [ ] Implementation Plan 中对每个媒体区域说明了实现方式（`<img>`、CSS background-div 等）及宽高比控制

好的媒体描述示例：
```
A full-width hero banner spanning the content column top, approximately 16:9 aspect ratio,
serving as the primary visual focus of the page.
```

不好的描述示例（不可泛化）：
```
A large gray placeholder box with a dashed border and centered "Image Placeholder" text overlay.
```

评分依据：
- ✅：所有媒体区域均以布局属性描述，无 placeholder 外观语言
- ⚠️：大部分描述用布局语言，但有轻微 placeholder 词汇（如 "placeholder" 单词出现，但重心在布局）
- ❌：描述以 placeholder 外观为主，或显式说明"this is a styled placeholder div"

---

### C — 颜色准确性

检查方法：将 reasoning 中的颜色值与 raw_html 的 CSS、outline 颜色信息交叉对比。

检查项：
- [ ] 主色调（背景色、主文字色、主题色）与代码一致
- [ ] 颜色值（hex / rgb）与 CSS 精确匹配或近似合理（±5 hex units 可接受）
- [ ] 无凭空捏造的颜色值（颜色在代码或截图中均无对应）

评分依据：
- ✅：全部颜色可在 raw_html 或 outline 中找到对应
- ⚠️：个别次要颜色有小偏差，主色正确
- ❌：主色调错误，或颜色值大量捏造

---

### H — 幻觉检测

检查项：
- [ ] 尺寸数值（px）与 outline visual_blocks 或 raw_html 中的数据一致（允许截图视觉估算，误差 ±20px 可接受）
- [ ] 元素数量（如"6个列表项"、"5个导航链接"）可在 raw_html 或 outline 中核实
- [ ] 具体文字内容（标题、链接文字）与 outline/raw_html 一致
- [ ] 不出现截图/代码中完全不存在的 UI 元素

评分依据：
- ✅：无可检测的事实性错误
- ⚠️：细节尺寸有轻微估算偏差，无法从截图精确验证
- ❌：明确的维度错误（数量级差异），或描述了代码/截图中不存在的元素

---

### A — Outline ↔ Reasoning 对齐

检查方法：核查 reasoning 是否正确使用了 outline 提供的结构信息，有无矛盾。

检查项：
- [ ] Layout 描述与 outline `major_structure` 一致
- [ ] 颜色描述来源于 outline `colors`，而非凭空
- [ ] outline 中的 layout_hints（flex row/grid）被正确引用
- [ ] outline 缺失的信息（如 logo 未被 visual_blocks 捕获）由截图补充，而非乱写

评分依据：
- ✅：Reasoning 与 outline 无矛盾，缺失信息由截图合理补充
- ⚠️：outline 有某项信息未被使用，但无矛盾
- ❌：Reasoning 与 outline 的结构描述直接矛盾

---

### I — 实现方案合理性

检查项：
- [ ] 布局方案（flexbox/grid）技术上正确且可实现
- [ ] 媒体区域有明确的 aspect ratio 控制方案（padding-bottom trick 或 `aspect-ratio` 属性）
- [ ] 色值直接可用（无 "approximately" 等导致代码歧义的表述）
- [ ] responsive breakpoint 有提及

评分依据：
- ✅：实现方案技术完整，可直接作为 HTML 编写指导
- ⚠️：部分细节不完整，但核心布局方案正确
- ❌：实现方案有技术错误（如用 float 实现 sticky sidebar 等）

---

### G — 泛化性

**从训练数据质量角度，评估该条 reasoning 是否能让模型在真实截图推理时泛化。**

核心问题：如果把这条 reasoning 替换为"同一 HTML 对应的真实网站截图"，模型学到的描述模式是否仍然适用？

检查项：
- [ ] 媒体区域描述不依赖 placeholder 外观（见 M 维度）
- [ ] 文字内容描述不依赖 HTML 中的 placeholder 文本标注（如"Image: Australia Day ceremony (filename)"）
- [ ] 没有显式提及训练数据特有的构造特征（如"ads/widgets have been removed"）
- [ ] 结构和布局术语（flex/grid/aspect-ratio）是通用的，不是特定于某种 placeholder 实现方式

评分依据：
- ✅：描述语言通用，适用于任何截图
- ⚠️：有轻微 placeholder 词汇但不是主要内容
- ❌：描述严重依赖 placeholder 外观或 HTML 文本标注，推理时无法复用

---

## 二、快速质检表格

使用如下表格汇总每条样本的评分：

| sample_id | F | L | M | C | H | A | I | G | 主要问题 |
|-----------|---|---|---|---|---|---|---|---|---------|
| xxx:0     |   |   |   |   |   |   |   |   |         |

整体评分标准（每维度 ✅=2 / ⚠️=1 / ❌=0）：
- ≥14 分：优秀（适合直接使用）
- 10-13 分：良好（可接受，建议注意 G 维度）
- <10 分：需修正

---

## 三、已知系统性问题与对应修复

| 问题 | 根因 | 修复状态 |
|---|---|---|
| nav 导航链接在 `navigation_links` 和 `lists` 双重出现 | outliner 未过滤 `<nav>` 内的 `<ul>` | ✅ 已修复（跳过 nav 内 ul） |
| 裸 tag 选择器（如 `a: horizontal flex row`）出现在 layout_hints | Playwright isVB 未过滤无 class/id 的元素 | ✅ 已修复（`_has_qualifier` 过滤） |
| `h1/h2` 等文本标签被标为 "full-width bar (likely nav/header)" | style_hint 触发条件未排除文本标签 | ✅ 已修复（`_TEXT_TAGS` 排除） |
| `ol`（面包屑）被标为 "full-width bar (likely nav/header)" | `_TEXT_TAGS` 未包含 `ol` | ✅ 已修复 |
| Logo placeholder div 未被 visual_blocks 捕获 | JS `STRUCT_RE` 无 logo/placeholder，`isVB` 文本阈值 15 过小 | ✅ 已修复 |
| reasoning 使用 placeholder 外观语言（dashed border 等） | 五路泄漏：outline 颜色/visual_blocks/raw_html CSS/HTML 文本/prompt 无禁用语 | ✅ 已修复（v3 四重清理） |
| `region_section_count` 对 `**bold:**` 格式计数为 0 | postprocess 正则未剥离 markdown 加粗符号 | ✅ 已修复 |
| `table`/`image` alignment warning 误报 | 关键词匹配太宽泛 | ✅ 已修复（移除这两个检查项） |
| 显式描述 placeholder 读取文本（"ads/widgets have been removed"） | 模型读取了 HTML 中的 placeholder 提示文字 | ✅ 已修复（`_clean_placeholder_text()` 清空文本） |
| Placeholder 重要 visual_block 被截断（sidebar-placeholder 被 pager 按钮占满 5 个槽） | visual_blocks `[:5]` 无优先级，重复块未去重 | ✅ 已修复（去重+placeholder 优先+上限 8） |
| 截图渲染中 dashed border / placeholder fill 仍可见 | Phase 1 截图前未对 placeholder CSS 中性化 | 🔄 已知限制，v4 方向：截图前替换 placeholder CSS 样式 |

---

## 四、执行频率建议

- 每次代码变更后（prompt/outliner/postprocess）：运行 6 条 demo 数据质检
- 每次大批量生产前：抽取 20 条进行质检
- 关键关注维度：M（媒体描述）和 G（泛化性），这两个维度直接影响模型推理质量
