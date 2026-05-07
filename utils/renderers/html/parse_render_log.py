#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析渲染日志文件，提取所有 WARN/ERROR 条目，
按数据 id 归组后写入 JSON，供后续统计分析和原数据过滤使用。

用法:
    python parse_render_log.py <log_file> [log_file ...] [-o output.json]

示例:
    python parse_render_log.py render.log -o issues.json
    python parse_render_log.py logs/*.log -o issues.json
"""

import re
import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path

# 匹配日志行: [LEVEL] | [CATEGORY] | id=log_id | k=v | ... | msg=...
_LINE_RE = re.compile(
    r'^\[(WARN|ERROR)\]\s*\|\s*\[([^\]]+)\]\s*\|\s*id=([^\s|]+)(.*)?$'
)
_FIELD_RE = re.compile(r'(\w+)=([^|]+)')


def _parse_fields(tail: str) -> dict:
    """解析行尾的 key=value 字段（含 msg=）"""
    fields = {}
    for m in _FIELD_RE.finditer(tail):
        key = m.group(1).strip()
        val = m.group(2).strip()
        # 尝试转数字
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        fields[key] = val
    return fields


def parse_log_file(path: str) -> list[dict]:
    """解析单个日志文件，返回所有 WARN/ERROR 条目列表。"""
    entries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n\r")
            m = _LINE_RE.match(line)
            if not m:
                continue
            level    = m.group(1)      # WARN / ERROR
            category = m.group(2)      # e.g. INFINITE_GROWTH
            log_id   = m.group(3)      # e.g. data_file:12345
            tail     = m.group(4) or ""

            # 从 log_id 拆分 source_file 和 item_id
            if ":" in log_id:
                source_file, item_id = log_id.rsplit(":", 1)
            else:
                source_file, item_id = "", log_id

            fields = _parse_fields(tail)

            entries.append({
                "log_id":      log_id,
                "source_file": source_file,
                "item_id":     item_id,
                "level":       level,
                "category":    category,
                "fields":      fields,
                "log_source":  Path(path).name,
                "lineno":      lineno,
            })
    return entries


def build_report(entries: list[dict]) -> dict:
    """
    按 log_id 归组，生成报告。

    输出结构:
    {
      "summary": {
        "total_flagged_ids": int,
        "total_entries": int,
        "by_level": {"WARN": int, "ERROR": int},
        "by_category": {category: int, ...},
        "warn_ids_count": int,
        "error_ids_count": int,
      },
      "ids": {
        "<log_id>": {
          "item_id":     str,
          "source_file": str,
          "has_error":   bool,
          "has_warn":    bool,
          "categories":  [str, ...],   // 去重、有序
          "entries": [
            {
              "level":    str,
              "category": str,
              "fields":   {k: v, ...},
              "log_source": str,
              "lineno":   int,
            },
            ...
          ]
        },
        ...
      }
    }
    """
    grouped: dict[str, dict] = {}

    for e in entries:
        lid = e["log_id"]
        if lid not in grouped:
            grouped[lid] = {
                "item_id":     e["item_id"],
                "source_file": e["source_file"],
                "has_error":   False,
                "has_warn":    False,
                "categories":  [],
                "entries":     [],
            }
        rec = grouped[lid]
        if e["level"] == "ERROR":
            rec["has_error"] = True
        else:
            rec["has_warn"] = True
        if e["category"] not in rec["categories"]:
            rec["categories"].append(e["category"])
        rec["entries"].append({
            "level":      e["level"],
            "category":   e["category"],
            "fields":     e["fields"],
            "log_source": e["log_source"],
            "lineno":     e["lineno"],
        })

    # 统计
    by_level: dict[str, int] = defaultdict(int)
    by_category_entries: dict[str, int] = defaultdict(int)   # 原始条目数（含重试）
    by_category_ids: dict[str, set] = defaultdict(set)       # 涉及的唯一 id 数
    warn_ids = set()
    error_ids = set()

    for e in entries:
        by_level[e["level"]] += 1
        by_category_entries[e["category"]] += 1

    for lid, rec in grouped.items():
        if rec["has_error"]:
            error_ids.add(lid)
        if rec["has_warn"]:
            warn_ids.add(lid)
        for cat in rec["categories"]:
            by_category_ids[cat].add(lid)

    # by_category 按条目数降序；同时记录对应 id 数，便于下游判断重试噪音
    sorted_cats = sorted(by_category_entries.keys(), key=lambda c: -by_category_entries[c])
    by_category = {
        c: {"entries": by_category_entries[c], "ids": len(by_category_ids[c])}
        for c in sorted_cats
    }

    summary = {
        "total_flagged_ids": len(grouped),
        "total_entries":     len(entries),
        "by_level":          dict(by_level),
        "by_category":       by_category,
        "warn_ids_count":    len(warn_ids),
        "error_ids_count":   len(error_ids),
    }

    return {"summary": summary, "ids": grouped}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="解析渲染日志，提取 WARN/ERROR 条目并写入 JSON"
    )
    parser.add_argument("logs", nargs="+", help="日志文件路径（支持多个）")
    parser.add_argument("-o", "--output", default="render_issues.json",
                        help="输出 JSON 文件路径（默认 render_issues.json）")
    args = parser.parse_args(argv)

    all_entries = []
    for log_path in args.logs:
        p = Path(log_path)
        if not p.exists():
            print(f"[SKIP] 文件不存在: {log_path}", file=sys.stderr)
            continue
        entries = parse_log_file(str(p))
        print(f"[PARSE] {p.name}: {len(entries)} 条 WARN/ERROR")
        all_entries.extend(entries)

    if not all_entries:
        print("[INFO] 未找到任何 WARN/ERROR 条目。")
        print("[NOTE] 若本次渲染跳过了已有 PNG 的条目（resume 模式），")
        print("       那些条目的历史问题不会出现在此日志中。")
        report = {"summary": {}, "ids": {}}
    else:
        report = build_report(all_entries)
        s = report["summary"]
        print(f"\n[SUMMARY]")
        print(f"  涉及 id 数量   : {s['total_flagged_ids']}")
        print(f"  总 log 条目数  : {s['total_entries']}  "
              f"(含同一 id 的重试条目)")
        print(f"  WARN 条目      : {s['by_level'].get('WARN', 0)}")
        print(f"  ERROR 条目     : {s['by_level'].get('ERROR', 0)}")
        print(f"  含 ERROR 的 id : {s['error_ids_count']}")
        print(f"  含 WARN 的 id  : {s['warn_ids_count']}")
        print(f"  各类别分布 (log条目数 / 涉及id数):")
        for cat, stat in s["by_category"].items():
            print(f"    {cat:<25} {stat['entries']:>4} 条 / {stat['ids']:>3} 个id")

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OUTPUT] 已写入: {out_path.resolve()}")


if __name__ == "__main__":
    main()
