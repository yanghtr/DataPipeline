# DataPipeline

## 项目目标

可扩展的数据管线：数据下载 → 格式转换 → 清洗 → 可视化。
当前阶段：下载模块

## 目录约定

Data/raw/        原始下载，按数据集名分子目录
Data/processed/  转换后的统一格式 JSONL (TODO)
Data/cleaned/    清洗后（TODO）

## 代码规范

- Python 3.11，所有函数必须有类型注解
- 日志统一用 loguru，避免裸 print

## 设计文档

下载模块设计背景：docs/requirements/downloader.md
