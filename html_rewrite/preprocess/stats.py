"""预处理统计数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MediaStats:
    total: int = 0
    images: int = 0
    videos: int = 0
    audios: int = 0
    iframes: int = 0      # 含 embed、object
    base64: int = 0
    regular: int = 0
    with_size: int = 0    # 成功获得宽高
    without_size: int = 0
    replaced: int = 0
    # 可选：fetch_media_size 开启时
    fetch_attempted: int = 0
    fetch_ok: int = 0
    fetch_failed: int = 0
    fetch_timeout: int = 0


@dataclass
class ScriptStats:
    external: int = 0
    inline_total: int = 0
    inline_truncated: int = 0
    inline_chars: list[int] = field(default_factory=list)


@dataclass
class JsonPayloadStats:
    total: int = 0
    truncated: int = 0
    chars: list[int] = field(default_factory=list)


@dataclass
class StyleStats:
    external_links: int = 0
    inline_total: int = 0
    inline_truncated: int = 0
    inline_chars: list[int] = field(default_factory=list)


@dataclass
class HiddenInputStats:
    total: int = 0
    truncated: int = 0
    value_chars: list[int] = field(default_factory=list)


@dataclass
class CommentStats:
    total: int = 0
    truncated: int = 0
    chars: list[int] = field(default_factory=list)


@dataclass
class FormatterStats:
    parse_ok: bool = True
    node_count_before: int = 0
    node_count_after: int = 0
    # 主要标签数量变化
    tag_counts_before: dict[str, int] = field(default_factory=dict)
    tag_counts_after: dict[str, int] = field(default_factory=dict)


@dataclass
class PreprocessStats:
    original_chars: int = 0
    cleaned_chars: int = 0
    visible_text_chars: int = 0

    media: MediaStats = field(default_factory=MediaStats)
    scripts: ScriptStats = field(default_factory=ScriptStats)
    json_payloads: JsonPayloadStats = field(default_factory=JsonPayloadStats)
    styles: StyleStats = field(default_factory=StyleStats)
    hidden_inputs: HiddenInputStats = field(default_factory=HiddenInputStats)
    comments: CommentStats = field(default_factory=CommentStats)
    formatter: FormatterStats = field(default_factory=FormatterStats)

    @property
    def compression_ratio(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round(self.cleaned_chars / self.original_chars, 4)

    def to_dict(self) -> dict:
        import dataclasses
        return {
            "original_chars": self.original_chars,
            "cleaned_chars": self.cleaned_chars,
            "visible_text_chars": self.visible_text_chars,
            "compression_ratio": self.compression_ratio,
            "media": dataclasses.asdict(self.media),
            "scripts": dataclasses.asdict(self.scripts),
            "json_payloads": dataclasses.asdict(self.json_payloads),
            "styles": dataclasses.asdict(self.styles),
            "hidden_inputs": dataclasses.asdict(self.hidden_inputs),
            "comments": dataclasses.asdict(self.comments),
            "formatter": dataclasses.asdict(self.formatter),
        }
