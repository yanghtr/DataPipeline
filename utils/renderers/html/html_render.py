#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML 渲染管线统一入口。

子命令:
  render   将 JSON 文件中的 html 字段批量渲染为 PNG 截图
  parse    解析渲染日志，提取 WARN/ERROR 条目，生成 issues.json
  filter   根据 issues.json 过滤原数据，剔除有问题的 id

典型工作流:
  # 1. 渲染，stderr 重定向到日志文件
  python html_render.py render \\
      --json_dir /data/json \\
      --files part_001.json part_002.json \\
      --images_dir /data/images \\
      --workers 16 \\
      2>render.log

  # 2. 解析日志，生成问题 id 报告
  python html_render.py parse render.log -o issues.json

  # 3. 过滤原数据，输出干净副本
  python html_render.py filter \\
      --issues issues.json \\
      --input /data/json \\
      --output /data/json_clean \\
      --mode error

每个子命令均支持 --help 查看完整参数列表。
"""

import sys


def _usage():
    print(__doc__.strip())
    print()
    print("用法: python html_render.py <command> [args ...]")
    print("      python html_render.py <command> --help")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _usage()
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    rest = sys.argv[2:]

    if command == "render":
        from render_html import main as _main
        _main(rest)
    elif command == "parse":
        from parse_render_log import main as _main
        _main(rest)
    elif command == "filter":
        from filter_by_issues import main as _main
        _main(rest)
    elif command == "preprocess":
        from preprocess_html import main as _main
        _main(rest)
    else:
        print(f"[ERROR] 未知子命令: {command!r}", file=sys.stderr)
        print("可用子命令: render / parse / filter / preprocess", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
