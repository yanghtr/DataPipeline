# utils 工具说明

## hf_downloader.py

从 HuggingFace 下载数据集、模型或 space，支持断点续传、多线程并行、高速下载和完整性校验。

### 安装依赖

```bash
pip install -r ../requirements.txt
```

### 快速使用

```bash
cd utils

# 下载数据集（默认类型）
python hf_downloader.py --repo-id allenai/c4

# 下载模型
python hf_downloader.py --repo-id meta-llama/Llama-2-7b --type model

# 指定线程数和输出目录
python hf_downloader.py --repo-id allenai/c4 --num-workers 8 --output-dir /data/raw

# 下载私有仓库
python hf_downloader.py --repo-id my-org/private-dataset --token hf_xxxxxxxx
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo-id` | 必填 | 仓库 ID，格式为 `owner/name` |
| `--type` | `dataset` | 仓库类型：`dataset` / `model` / `space` |
| `--output-dir` | `../Data/raw` | 输出根目录，文件保存至 `<output-dir>/<repo-name>/` |
| `--num-workers` | `16` | 并行下载线程数，网络带宽充足时可调高 |
| `--token` | 无 | HuggingFace Access Token，访问私有仓库时必填 |
| `--no-verify` | 关闭 | 加此参数跳过下载后的完整性校验 |

### 断点续传

中断后直接重新运行相同命令即可，程序会自动检测本地已有文件并跳过。

### 日志

运行日志自动写入 `logs/hf_downloader_<时间>.log`，单文件上限 100MB，保留 7 天。
