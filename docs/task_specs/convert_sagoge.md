# SAgoge 数据转换任务说明

## 1. 任务目标

请将指定的 SAgoge 数据子集转换为统一多模态数据格式。

统一输出格式必须严格遵守：

- `.claude/skills/multimodal-data-format/SKILL.md`

本文件只描述**这一次具体数据集转换任务**的要求，不定义通用格式本身。

---

## 2. 本次任务范围

本次只处理以下数据范围：

- 数据集名称：`SAgoge`
- 原始数据集根目录：`/home/yanghaitao/Projects/Data/raw/SAgoge/`
- 处理完后数据集根目录：`/home/yanghaitao/Projects/Data/processed/SAgoge/`
- 输入文件相对路径（相对原始数据集根目录）：
  - stage1/icon/generation/img2svg/img2svg_alpaca.jsonl
  - stage1/icon/generation/text2svg/text2svg_alpaca.jsonl
  - stage2/icon/generation/img2svg/img2svg_alpaca.jsonl
  - stage2/icon/generation/text2svg/text2svg_alpaca.jsonl
  - stage2/illustration/img2svg/img2svg_alpaca.jsonl
  - stage2/illustration/text2svg/text2svg_alpaca.jsonl
- 输出文件路径：文件存储在处理完后数据集根目录，相对目录层级和输入文件相对路径一致，即：（相对处理完后数据集根目录）
  - stage1/icon/generation/img2svg/data_000000.jsonl
  - stage1/icon/generation/text2svg/data_000000.jsonl
  - stage2/icon/generation/img2svg/data_000000.jsonl
  - stage2/icon/generation/text2svg/data_000000.jsonl
  - stage2/illustration/img2svg/data_000000.jsonl
  - stage2/illustration/text2svg/data_000000.jsonl

---

## 3. 输入数据格式

### 3.1 输入文件格式

本次输入文件格式为 jsonl

### 3.2 单条原始样本示例

示例1:

```json
{"instruction": "You are a professional in vector-based image generation. Please produce an SVG that matches the following image and instruction.\n Image: <image>\n Instruction: Generate a red five-pointed star.", "input": "", "output": "<svg width=\"128\" height=\"128\" viewBox=\"0 0 128 128\"><path fill=\"#de0000\" d=\"M0 49.65h128L24.46 124.89 64.05 3.11l39.59 121.78z\"/></svg>", "images": ["icon/fb9cc147-5362-42bf-862d-b30f64305272.png"]}
```
- user text 来源字段：`Instruction: ` 后面的`Generate a red five-pointed star.`
- assistant text 来源字段：`"output"`这个key后面的内容，注意对于SVG代码应该用markdown格式包裹，即````svg\nSVG_CODE_HERE\n````
- user image 来源字段：`"images"`这个key后面的内容，注意要求是文本，不能是list。验证这个list长度就是1，然后直接取第0个就行; 如果是空，则设成`""`


示例2:

```json
{"instruction": "As an expert in SVG generation, you will be given an instruction. Please generate an SVG that satisfies the instruction.\n Instruction: \"Draw a black diagonal line from the top left to the bottom right of a white square.\"", "input": "", "output": "<svg width=\"128\" height=\"128\" viewBox=\"0 0 128 128\"><path d=\"M65.85 128H36.57L62.15 0h29.28z\"/></svg>", "images": []}
```
- user text 来源字段：`Instruction: ` 后面的`Draw a black diagonal line from the top left to the bottom right of a white square.`，注意把前后无用的空格和`\"`去掉。
- assistant text 来源字段：`"output"`这个key后面的内容，注意对于SVG代码应该用markdown格式包裹，即````svg\nSVG_CODE_HERE\n````
- user image 来源字段：`"images"`这个key后面的内容，注意要求是文本，不能是list。如果是空，则设成`""`


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

现在先默认：
- `width = 0`
- `height = 0`

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

