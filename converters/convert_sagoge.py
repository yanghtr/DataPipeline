"""
SAgoge 数据集转换脚本

将 SAgoge alpaca 格式 JSONL 转换为统一多模态 canonical schema 格式。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Canonical schema builders
# ---------------------------------------------------------------------------

def build_text_item(text: str) -> dict:
    return {
        "type": "text",
        "text": {
            "type": "string",
            "format": "utf-8",
            "string": text,
        },
    }


def build_image_item(
    relative_path: str,
    image_format: str,
    width: int = 0,
    height: int = 0,
) -> dict:
    return {
        "type": "image",
        "image": {
            "type": "relative_path",
            "format": image_format,
            "relative_path": relative_path,
            "width": width,
            "height": height,
        },
    }


def build_sample(
    user_content: list[dict],
    assistant_text: str,
    train_mode: str,
) -> dict:
    meta_prompt_map = {
        "sft": [""],
        "pretrain": ["", ""],
    }
    meta_prompt = meta_prompt_map[train_mode.lower()]
    return {
        "meta_prompt": meta_prompt,
        "data": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [build_text_item(assistant_text)],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Canonical schema validator
# ---------------------------------------------------------------------------

VALID_META_PROMPTS = ([""], ["", ""])
VALID_IMAGE_FORMATS = {"image/jpeg", "image/png"}


def validate_sample(sample: dict) -> tuple[bool, Optional[str]]:
    if "meta_prompt" not in sample:
        return False, "missing meta_prompt"
    if "data" not in sample:
        return False, "missing data"
    if not isinstance(sample["meta_prompt"], list):
        return False, "meta_prompt must be list"
    if sample["meta_prompt"] not in VALID_META_PROMPTS:
        return False, f"invalid meta_prompt: {sample['meta_prompt']}"
    if not isinstance(sample["data"], list) or len(sample["data"]) != 2:
        return False, "data must be list of length 2"

    for turn_idx, turn in enumerate(sample["data"]):
        role = turn.get("role")
        content = turn.get("content")
        expected_role = "user" if turn_idx == 0 else "assistant"
        if role != expected_role:
            return False, f"data[{turn_idx}].role must be '{expected_role}'"
        if not isinstance(content, list) or len(content) == 0:
            return False, f"data[{turn_idx}].content must be non-empty list"
        for item in content:
            ok, reason = _validate_item(item, allow_image=(turn_idx == 0))
            if not ok:
                return False, f"data[{turn_idx}] item error: {reason}"
    return True, None


def _validate_item(item: dict, allow_image: bool) -> tuple[bool, Optional[str]]:
    item_type = item.get("type")
    if item_type == "text":
        t = item.get("text", {})
        if t.get("type") != "string":
            return False, "text.type must be 'string'"
        if t.get("format") != "utf-8":
            return False, "text.format must be 'utf-8'"
        if not isinstance(t.get("string"), str):
            return False, "text.string must be str"
        return True, None
    elif item_type == "image":
        if not allow_image:
            return False, "image item not allowed in assistant turn"
        img = item.get("image", {})
        if img.get("type") != "relative_path":
            return False, "image.type must be 'relative_path'"
        if img.get("format") not in VALID_IMAGE_FORMATS:
            return False, f"image.format must be one of {VALID_IMAGE_FORMATS}"
        if not isinstance(img.get("relative_path"), str) or not img["relative_path"]:
            return False, "image.relative_path must be non-empty str"
        if not isinstance(img.get("width"), int):
            return False, "image.width must be int"
        if not isinstance(img.get("height"), int):
            return False, "image.height must be int"
        return True, None
    else:
        return False, f"unknown item type: {item_type!r}"


# ---------------------------------------------------------------------------
# SAgoge-specific parsing
# ---------------------------------------------------------------------------

_INSTRUCTION_PATTERN = re.compile(
    r'Instruction:\s*"?(.*?)"?\s*$',
    re.DOTALL,
)

_EXT_TO_FORMAT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _extract_user_text(instruction: str) -> Optional[str]:
    """提取 'Instruction: ' 后面的 user text，去除首尾空格和多余引号。"""
    m = _INSTRUCTION_PATTERN.search(instruction)
    if not m:
        return None
    text = m.group(1).strip().strip('"').strip()
    return text if text else None


def _image_has_before_instruction(instruction: str) -> bool:
    """判断 <image> 标记是否出现在 'Instruction:' 之前。"""
    img_pos = instruction.find("<image>")
    ins_pos = instruction.find("Instruction:")
    if img_pos == -1 or ins_pos == -1:
        return False
    return img_pos < ins_pos


def _get_image_format(relative_path: str) -> Optional[str]:
    ext = Path(relative_path).suffix.lower()
    return _EXT_TO_FORMAT.get(ext)


@dataclass
class ConversionStats:
    total: int = 0
    success: int = 0
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def record_skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def report(self) -> str:
        lines = [
            f"总输入样本数:      {self.total}",
            f"成功转换样本数:    {self.success}",
            f"跳过样本数:        {self.skipped}",
        ]
        if self.skip_reasons:
            lines.append("跳过原因明细:")
            for reason, count in sorted(self.skip_reasons.items()):
                lines.append(f"  {reason}: {count}")
        return "\n".join(lines)


def convert_record(
    raw: dict,
    train_mode: str,
    log_prefix: str = "",
) -> tuple[Optional[dict], Optional[str]]:
    """
    将一条 SAgoge alpaca 记录转换为 canonical schema。

    返回 (sample, skip_reason)：
    - 成功时 skip_reason 为 None
    - 失败时 sample 为 None，skip_reason 为跳过原因字符串
    """
    instruction: str = raw.get("instruction", "")
    output: str = raw.get("output", "")
    images: list = raw.get("images", [])

    # --- assistant text ---
    if not output or not output.strip():
        return None, "missing_assistant_text"
    assistant_text = f"```svg\n{output}\n```"

    # --- user text ---
    user_text = _extract_user_text(instruction)
    if not user_text:
        return None, "missing_user_content"

    # --- user image ---
    has_image_tag = "<image>" in instruction
    if has_image_tag:
        if not isinstance(images, list) or len(images) != 1:
            return None, "invalid_image_format"
        relative_path = images[0]
        if not isinstance(relative_path, str) or not relative_path:
            return None, "invalid_relative_path"
        # 防止绝对路径混入
        if Path(relative_path).is_absolute():
            return None, "invalid_relative_path"
        image_format = _get_image_format(relative_path)
        if image_format is None:
            return None, "invalid_image_format"
        image_item = build_image_item(relative_path, image_format)
        text_item = build_text_item(user_text)
        # 顺序：<image> 在 Instruction 之前 → image first
        if _image_has_before_instruction(instruction):
            user_content = [image_item, text_item]
        else:
            user_content = [text_item, image_item]
    else:
        # text-only：不伪造 image item
        if images:
            logger.warning(
                f"{log_prefix} instruction 无 <image> 标记但 images 非空，忽略图片"
            )
        user_content = [build_text_item(user_text)]

    sample = build_sample(user_content, assistant_text, train_mode)

    ok, reason = validate_sample(sample)
    if not ok:
        return None, f"schema_validation_failed: {reason}"

    return sample, None


# ---------------------------------------------------------------------------
# File-level conversion
# ---------------------------------------------------------------------------

def convert_file(
    input_path: Path,
    output_path: Path,
    train_mode: str,
    image_root: Optional[Path],
    log_path: Optional[Path],
) -> ConversionStats:
    stats = ConversionStats()

    # 配置日志
    log_handlers: list[dict] = [{"sink": sys.stderr, "level": "INFO"}]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append({"sink": str(log_path), "level": "DEBUG"})
    logger.configure(handlers=log_handlers)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            stats.total += 1
            prefix = f"[{input_path.name}:{lineno}]"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"{prefix} JSON 解析失败: {e}")
                stats.record_skip("json_parse_error")
                continue

            sample, skip_reason = convert_record(raw, train_mode, log_prefix=prefix)
            if skip_reason:
                logger.debug(f"{prefix} 跳过: {skip_reason}")
                stats.record_skip(skip_reason)
                continue

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats.success += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 SAgoge alpaca JSONL 转换为统一多模态 canonical schema。"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL 文件路径")
    parser.add_argument(
        "--train-mode",
        choices=["sft", "pretrain"],
        default="sft",
        help="训练模式，默认 sft",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="图片根目录（可选，用于将来读取宽高等信息）",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="日志文件路径（可选）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)

    logger.info(f"开始转换: {input_path} -> {output_path}")
    stats = convert_file(
        input_path=input_path,
        output_path=output_path,
        train_mode=args.train_mode,
        image_root=args.image_root,
        log_path=args.log_path,
    )

    report = stats.report()
    logger.info(f"\n{report}")
    print(report)


if __name__ == "__main__":
    main()
