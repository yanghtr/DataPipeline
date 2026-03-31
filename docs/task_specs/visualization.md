# 数据可视化任务说明

写一个数据可视化网站来可视化来可视化多模态数据。数据格式遵守统一 skill 中定义的 canonical schema。涵盖内容如下：

- Reads JSONL (one JSON object per line).
- Scans schema from the full JSONL (keys/roles/content types/images/videos hints).
- Supports two sampling modes for sample_n:
  - random_sample = off: use first N rows
  - random_sample = on: random sample N rows
- Resolves image full path with --image_root + relative path from each row.
- Graceful fallback: if image loading fails, shows a gray placeholder image with the full path and error hint.
- panel includes:
  - sample selector with line/global_id/current_index
  - raw JSON
  - parsed messages
  - input image(s)
  - online renderer of SVG/HTML
  - question
  - answer/code

