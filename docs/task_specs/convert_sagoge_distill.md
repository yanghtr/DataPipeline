# SAgoge 蒸馏数据转换任务说明

## 1. 任务目标

请将指定的数据集转换为统一多模态数据格式。

统一输出格式必须严格遵守：

- `.claude/skills/multimodal-data-format/SKILL.md`

本文件只描述**这一次具体数据集转换任务**的要求，不定义通用格式本身。

---

## 2. 本次任务范围

本次只处理以下数据范围：

- 原始数据集文件：`/home/yanghaitao/Projects/DataPipeline/outputs/distillation/SAgoge/svg_responses.jsonl`
- 处理完后数据集根目录：`/home/yanghaitao/Projects/Data/processed/SAgoge_distillation/`
- 输出文件路径：文件存储在处理完后数据集根目录，相对目录层级和输入文件相对路径一致，即：（相对处理完后数据集根目录）
  - jsonl根目录：stage1_icon/img2svg/jsonl/data_000000.jsonl
  - 图片根目录：stage1_icon/img2svg/images/

实际代码中，以上路径都可以手动配置

---

## 3. 输入数据格式

### 3.1 输入文件格式

本次输入文件格式为 jsonl

### 3.2 单条原始样本示例

示例1:

```json
{"id": "stage1_icon/img2svg/data_000000:1200337", "instruction": "A white background with no discernible shapes or elements.", "messages": [{"role": "system", "content": "You are an expert SVG generator. Generate clean, valid, self-contained, renderable SVG that matches the user's instruction. Focus on semantic accuracy and standards-compliant output."}, {"role": "user", "content": [{"type": "text", "text": "Design a scalable vector graphic (SVG) for the following instruction:\n\nA white background with no discernible shapes or elements.\n\nConstraints:\n- The SVG must be self-contained and directly renderable.\n- Use standard XML/SVG syntax and include an explicit viewBox.\n- Include the root <svg> element with xmlns=\"http://www.w3.org/2000/svg\".\n- Do not use script, foreignObject, external images, external fonts, external stylesheets, or external references.\n- Return exactly one complete SVG inside a single ```svg ... ``` code block and no other text.\n- The SVG inside the code block must start with <svg and end with </svg>."}]}], "response": "thought\n*   Goal: An SVG with a white background and no discernible shapes or elements.\n    *   Constraints:\n        *   Self-contained, renderable.\n        *   Standard XML/SVG syntax, explicit `viewBox`.\n        *   Root `<svg>` element with `xmlns=\"http://www.w3.org/2000/svg\"`.\n        *   No scripts, foreignObjects, external assets.\n        *   One code block, no other text.\n\n    *   An SVG with a white background can be achieved by:\n        *   Defining the background color in the CSS of the svg or by adding a white rectangle that fills the entire viewBox.\n        *   Alternatively, since SVGs have a transparent background by default, a white rectangle is the most explicit way to ensure the background is white regardless of the viewer's environment.\n    *   \"No discernible shapes or elements\" means no other icons, lines, or figures.\n\n    *   `<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\" viewBox=\"0 0 100 100\">`\n    *   `<rect width=\"100\" height=\"100\" fill=\"white\" />`\n    *   `</svg>`\n\n    *   White background? Yes (rect fill=\"white\").\n    *   No discernible shapes/elements? Yes (only the background).\n    *   Self-contained/renderable? Yes.\n    *   Standard syntax/viewBox? Yes.\n    *   No prohibited elements? Yes.\n    *   One code block? Yes.```svg\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\" width=\"100\" height=\"100\">\n  <rect width=\"100\" height=\"100\" fill=\"white\" />\n</svg>\n```", "model": "gemma-4-31b-it", "prompt_tokens": 210, "completion_tokens": 443, "finish_reason": "stop"}
```
- user text 来源字段：converters/templates/image_to_svg_templates.txt 里面的prompt。注意里面有50条，我们全局设一个seed来随机采样，确保后续可以重新浮现。
- assistant text 来源字段：`"response"`这个key后面的内容，注意对于SVG代码应该用markdown格式包裹，即````svg...````。注意这里的思考过程可能有多个`<svg ...></svg>`，应该是取最后一个SVG代码
- user image 来源字段：从0开始对数据做编号，编号可以用来作为图片的`relative_path` （相对于输出文件目录中的图片根目录）。实际图片由SVG代码渲染得到
- 最后新增一个 "_meta" 字段(和"meta_prompt", "data"字段同一级)：加入`"instruction"` 这个子字段，来源于单条原始样本的`"instruction"`字段。"_meta"字段后续会用来记录一些中间的结果


---

## 4. 输出目标

输出为**统一 canonical schema** 格式，每条样本对应一个标准化后的 JSON 对象。

### 4.1 输出文件格式

- 输出文件格式：jsonl
- 编码：`UTF-8`
- 每行一条合法 sample

### 4.2 训练模式

本次数据属于哪种训练模式：`SFT`, `meta_prompt = [""]`.

### 4.3 推荐顺序

和原始数据一致，如果先`<image>`然后才是`Instruction`后面的user text，则是image在前。

### 4.4 宽高规则

根据SVG代码来定。比如`<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\" width=\"100\" height=\"100\">`，如果存在`width`和`height`，则使用该宽高。如果不存在，则使用`viewBox`里面的大小，比如`viewBox=\"0 0 100 100\"`则使用`width=100` 以及`height=100`。确定宽高后，我们需要使用CairoSVG对SVG代码进行并行渲染得到图片，然后还要读取图片做一下校验

### 4.5 SVG渲染

根据前面定的宽高规则得到宽高，用来渲染SVG代码。图片存在输出文件路径里面的图片根目录下，命名可以根据编号定。我们需要使用并行渲染。
我们需要在对应的jsonl文件的每一行加入该数据的meta info，由 `"_meta": {}` 这个字段记录。该字段记录数据的相关信息，比如输入数据的`response`, `model`，以及SVG代码是否符合格式，渲染图片是否成功，渲染图片是不是全白等。这个中间jsonl存在类似`stage1_icon/img2svg/xxx/xxx.jsonl`的文件里面。最后跑完filter模式过滤完后才存到最后的 stage1_icon/img2svg/jsonl/data_000000.jsonl (此时需要丢掉 _meta 字段)

### 4.6 简单筛选

代码还需要支持一种filter模式，在转完数据格式之后可以再跑filter模式快速筛选数据。筛选数据时可以直接根据已经记录好的 _meta 字段，筛选掉不符合SVG代码格式，不能正确解析、渲染或者渲染全白的数据.

---

## 5 CLI 参数建议

至少支持：

- `--input`
- `--output`
- `--train-mode`
- `--image-root`（可以给后面帮忙得到完整图片路径来读取宽高信息）
- `--log-path`（可选）

---

## 6. 推荐输出统计

建议代码最终打印或保存以下统计信息：

- 总输入样本数
- 成功转换样本数
- 跳过样本数
- 各类 skip reason 的计数

例如：

- missing_user_content
- missing_assistant_text
- invalid_image_format
- invalid_relative_path
- schema_validation_failed

---

## 7. 对 Claude Code 的明确要求

请严格遵守以下要求生成代码：

1. 严格遵守统一 skill 中定义的 canonical schema
2. 不要自行猜测字段含义，必须只按本文件的字段映射实现
3. 不要把绝对路径写入 `image.relative_path`
4. assistant 仍然保持 text-only
5. text-only 数据不能伪造 image item
6. image-only 数据不能伪造 text item
7. 代码尽量模块化，便于未来复用
8. 默认非法样本跳过并记录日志
9. 给出使用示例。

