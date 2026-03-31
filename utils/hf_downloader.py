import os

# 必须在 import huggingface_hub 之前设置，否则 hf_transfer 不生效
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import importlib.util
import argparse
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import snapshot_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
from loguru import logger


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class DownloadConfig:
    repo_id: str
    dataset_type: str           # "dataset" | "model" | "space"
    output_dir: Path
    num_workers: int = 16
    use_symlinks: bool = False
    token: str | None = None


# ---------------------------------------------------------------------------
# 下载器
# ---------------------------------------------------------------------------

class HFDownloader:
    def __init__(self, config: DownloadConfig) -> None:
        self.config = config
        self._setup_logger()
        self._setup_env()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _setup_logger(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "hf_downloader_{time}.log",
            rotation="100 MB",
            retention="7 days",
            encoding="utf-8",
        )

    def _setup_env(self) -> None:
        """检测 hf_transfer 安装状态，给出明确提示。"""
        if importlib.util.find_spec("hf_transfer") is not None:
            logger.info("hf_transfer 已启用，将使用高速下载模式")
        else:
            logger.warning(
                "hf_transfer 未安装，使用标准下载速度。"
                "可通过 `pip install hf_transfer` 安装以提升速度"
            )

    def _resolve_output_dir(self) -> Path:
        """按 Data/raw/<repo_name>/ 规则构建输出目录并创建。"""
        repo_name = self.config.repo_id.split("/")[-1]
        target = self.config.output_dir / repo_name
        target.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {target.resolve()}")
        return target

    def _get_existing_files(self, local_dir: Path) -> set[str]:
        """递归扫描本地目录，返回相对路径集合（用于断点续传）。"""
        if not local_dir.exists():
            return set()
        existing = {
            str(p.relative_to(local_dir))
            for p in local_dir.rglob("*")
            if p.is_file()
        }
        return existing

    def _get_remote_files(self) -> list[str]:
        """获取远端仓库的完整文件列表。"""
        return list(
            list_repo_files(
                self.config.repo_id,
                repo_type=self.config.dataset_type,
                token=self.config.token,
            )
        )

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def download(self) -> Path:
        """执行下载，返回本地目录路径。"""
        local_dir = self._resolve_output_dir()

        # 断点续传：计算已完成文件，加入 ignore_patterns 跳过
        existing = self._get_existing_files(local_dir)
        try:
            remote_files = self._get_remote_files()
        except (RepositoryNotFoundError, EntryNotFoundError) as e:
            logger.error(f"无法访问仓库 {self.config.repo_id}: {e}")
            raise

        skip_files = [f for f in remote_files if f in existing]
        if skip_files:
            logger.info(f"断点续传：跳过已完成文件 {len(skip_files)} 个，共 {len(remote_files)} 个")
        else:
            logger.info(f"全新下载，共 {len(remote_files)} 个文件")

        logger.info(
            f"开始下载 {self.config.repo_id} "
            f"[type={self.config.dataset_type}, workers={self.config.num_workers}]"
        )

        snapshot_download(
            repo_id=self.config.repo_id,
            repo_type=self.config.dataset_type,
            local_dir=str(local_dir),
            local_dir_use_symlinks=self.config.use_symlinks,
            ignore_patterns=skip_files if skip_files else None,
            max_workers=self.config.num_workers,
            token=self.config.token,
        )

        logger.info(f"下载完成: {local_dir.resolve()}")
        return local_dir

    def verify(self, local_dir: Path) -> bool:
        """对比远端文件列表与本地文件，返回是否完整。"""
        logger.info("开始校验文件完整性...")
        try:
            remote_files = set(self._get_remote_files())
        except Exception as e:
            logger.error(f"获取远端文件列表失败，跳过校验: {e}")
            return False

        local_files = self._get_existing_files(local_dir)
        missing = remote_files - local_files

        if missing:
            logger.error(f"校验失败，缺失 {len(missing)} 个文件: {sorted(missing)}")
            return False

        logger.info(f"校验通过，共 {len(local_files)} 个文件")
        return True


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _parse_args() -> DownloadConfig:
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载数据集 / 模型 / space"
    )
    parser.add_argument("--repo-id", required=True, help='仓库 ID，如 "allenai/c4"')
    parser.add_argument(
        "--type",
        dest="dataset_type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="仓库类型（默认: dataset）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Data/raw"),
        help="输出根目录（默认: ../Data/raw）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="并行下载线程数（默认: 16）",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace Access Token（私有仓库必填）",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过下载后的完整性校验",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = _parse_args()
    config = DownloadConfig(
        repo_id=args.repo_id,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        token=args.token,
    )

    downloader = HFDownloader(config)
    local_dir = downloader.download()

    if not args.no_verify:
        ok = downloader.verify(local_dir)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
