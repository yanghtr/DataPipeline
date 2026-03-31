下载模块需求文档

- 背景

这个文件记录 utils/hf_downloader.py 的设计意图，供 Claude 理解为什么这么设计。

- 核心需求
从 HuggingFace 下载数据集，要求：

    - 参数：repo_id（如 "allenai/c4"）、dataset_type（dataset / model / space）
    - 自动断点续传：中断后重跑自动跳过已下载文件
    - 多线程并行：默认 16 线程，可通过 --num-workers 调整
    - 快速下载：尽可能快下载，如使用hf_transfer
    - 使用snapshot_download，该设置的环境变量需要设置正确
    - 自动日志：下载进度、速度、ETA
    - 指定输出路径，默认不用symlink
    - 下载完后校验文件都下载成功

关键设计决策

用 huggingface_hub.snapshot_download 而不是 datasets 库，更通用
HF_HUB_ENABLE_HF_TRANSFER=1 环境变量必须在调用前设置
断点续传通过 local_dir_use_symlinks=False + 检查已有文件实现

调用方式
```bash
cd utils
export HF_ENDPOINT=https://hf-mirror.com
python hf_downloader.py --repo-id xxx --type dataset --num-workers 8
```
