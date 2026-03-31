# 多模态数据格式定义 Skill

## 目标

本 Skill 用于定义数据管线中的**统一样本格式（canonical format）**，供不同数据集的转换脚本、清洗脚本、筛选脚本、蒸馏脚本复用。 这个 Skill **只负责格式定义**。

换句话说：

- **这个 Skill 定义“输出长什么样”**
- 不同数据集各自的脚本负责“怎么转成这个样子”

---

## 适用范围

当前版本只考虑最简单、最常见的场景：

- 只支持 **1 个 user + 1 个 assistant**
- assistant 只输出 **text**
- user 可以是：
  - 纯文本
  - 图像 + 文本
  - 纯图像

其中，**纯文本样本**就是 user 的 `content` 里没有 image item。

---

## 统一格式总览

每条样本必须是一个 JSON object，顶层结构如下：

```json
{
  "meta_prompt": [""],
  "data": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": {
            "type": "relative_path",
            "format": "image/jpeg",
            "relative_path": "images_path/image_1.jpg",
            "width": 0,
            "height": 0
          }
        },
        {
          "type": "text",
          "text": {
            "type": "string",
            "format": "utf-8",
            "string": "请描述这张图片。"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": {
            "type": "string",
            "format": "utf-8",
            "string": "图片里面有..."
          }
        }
      ]
    }
  ]
}
```

---

## 顶层字段定义

### 1. `meta_prompt`

类型：`list[str]`

必须存在。

当前约定：

- **SFT 数据**：
  ```json
  [""]
  ```
- **预训练数据**：
  ```json
  ["", ""]
  ```

说明：

- 这里不在本 Skill 中解释多个 `meta_prompt` 的训练语义
- 这里只定义格式约定，具体含义由下游训练代码决定

### 2. `data`

类型：`list[dict]`

必须存在，且当前版本必须**长度恰好为 2**：

- `data[0]`：user turn
- `data[1]`：assistant turn

---

## `data` 内部结构定义

### 1. user turn

固定格式：

```json
{
  "role": "user",
  "content": [...]
}
```

要求：

- `role` 必须是 `"user"`
- `content` 必须是非空列表
- `content` 中允许的 item 类型只有：
  - `image`
  - `text`

### 2. assistant turn

固定格式：

```json
{
  "role": "assistant",
  "content": [...]
}
```

要求：

- `role` 必须是 `"assistant"`
- `content` 必须是非空列表
- 当前版本中 assistant 的 `content` **只允许 text item**

---

## `content` item 类型定义

当前版本只支持两类 item：

- `text`
- `image`

---

## Text Item 定义

标准格式：

```json
{
  "type": "text",
  "text": {
    "type": "string",
    "format": "utf-8",
    "string": "具体文本内容"
  }
}
```

### 字段要求

#### 顶层

- `type` 必须是 `"text"`

#### `text` 对象内部

- `type` 必须是 `"string"`
- `format` 必须是 `"utf-8"`
- `string` 必须是字符串

### 说明

- 不允许把文本直接写成裸字符串
- 必须包装成统一的 `text` object
- 空字符串是否允许，由具体数据转换脚本决定；但一般不建议保留空文本

---

## Image Item 定义

标准格式：

```json
{
  "type": "image",
  "image": {
    "type": "relative_path",
    "format": "image/jpeg",
    "relative_path": "images_path/image_1.jpg",
    "width": 0,
    "height": 0
  }
}
```

### 字段要求

#### 顶层

- `type` 必须是 `"image"`

#### `image` 对象内部

- `type` 必须是 `"relative_path"`
- `format` 只允许：
  - `"image/jpeg"`
  - `"image/png"`
- `relative_path` 必须是非空字符串
- `width` 必须是整数
- `height` 必须是整数

### `relative_path` 的语义

这里的图像路径**不是绝对路径**，而是相对于外部 `image_root` 的相对路径。

下游读取图像时，完整路径按如下方式拼接：

```python
full_path = image_root / relative_path
```

说明：

- `image_root` 在样本外部定义
- 不同数据集可以使用不同的 `image_root`
- `relative_path` 的具体生成规则由各个数据集自己的 converter 脚本决定
- 统一格式层只要求最终写出来的是 `relative_path`

### `width` / `height` 的语义

- 如果已知真实尺寸，可以填写真实值
- 如果当前阶段不方便获取，可统一写 `0`
- `0` 表示未知或未填充

---

## 顺序约定

user `content` 的顺序需要根据外部指令进行指定，可以先放 image item，也可以先放 text item，取决于不同数据集。

---

## 校验规则

一个合法样本必须满足以下全部条件。

### 顶层校验

1. 必须有 `meta_prompt`
2. 必须有 `data`
3. `meta_prompt` 必须是 list
4. `data` 必须是 list
5. `data` 长度必须等于 2

### `meta_prompt` 校验

当前版本只允许：

- `[""]`
- `["", ""]`

### `data[0]` 校验

1. `role == "user"`
2. `content` 是非空 list
3. `content` 内每个 item 必须是合法的 `text` 或 `image`

### `data[1]` 校验

1. `role == "assistant"`
2. `content` 是非空 list
3. `content` 内每个 item 必须是合法的 `text`
4. 不允许 assistant `content` 中出现 image item

### `text` item 校验

1. 顶层 `type == "text"`
2. 必须存在 `text` 对象
3. `text.type == "string"`
4. `text.format == "utf-8"`
5. `text.string` 必须是字符串

### `image` item 校验

1. 顶层 `type == "image"`
2. 必须存在 `image` 对象
3. `image.type == "relative_path"`
4. `image.format` 必须是 `image/jpeg` 或 `image/png`
5. `image.relative_path` 必须是非空字符串
6. `image.width` 必须是整数
7. `image.height` 必须是整数

---

## 推荐的构造函数接口（可选）

这个 Skill 不强制代码结构，但如果后续需要共享 builder / validator，可以参考如下接口：

```python
build_text_item(text: str) -> dict
build_image_item(relative_path: str, image_format: str, width: int = 0, height: int = 0) -> dict
build_sample(user_content: list[dict], assistant_text: str, train_mode: str) -> dict
validate_sample(sample: dict) -> tuple[bool, str | None]
```

其中：

- `train_mode` 只支持：
  - `"sft"`
  - `"pretrain"`

---

## 对 Claude Code / 其他代码生成工具的要求

当使用本 Skill 为不同数据集编写转换脚本时，生成的代码必须遵守：

1. 严格输出本 Skill 定义的统一格式
2. 不把绝对路径写进 `relative_path`
3. text-only 数据通过“省略 image item”表示，而不是写空 image
4. assistant 在 v1 中必须保持 text-only
5. 对样本做格式校验

---

## TODO（留给后续版本）

### 1. 多轮对话

TODO：支持多于 1 轮的 user / assistant 对话。

### 2. 多图输入

TODO：支持 user 中出现多个 image item，并明确它们的顺序和语义。

### 3. assistant 多模态输出

TODO：支持 assistant 输出 image / structured data / tool call。

### 4. 视频/音频

TODO：扩展更多 modality 类型。

### 5. 宽高是否强制填写

TODO：后续是否要求所有 converter 尽量填真实 `width` / `height`。
