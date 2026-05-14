from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.cache_manager import CacheManager
from src.data.dataset import NpzSkeletonDataset, SkeletonDataset, safe_collate
from src.data.samplers import SamplingStrategy
from src.utils.config_utils import (
    apply_overrides,
    load_config,
    prepare_run_dirs,
    save_config,
)
from src.utils.distributed import get_world_size, is_main_process, setup_distributed
from src.utils.logging_utils import log_config_summary, setup_logger
from src.utils.seed import seed_everything
from src.utils.wandb_utils import init_wandb


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value.")
    parser.add_argument("--wandb_mode", default=None, help="offline, online, or disabled.")
    parser.add_argument("--eval_during_train", action="store_true", help="Enable validation during training.")
    parser.add_argument("--eval_freq", type=int, default=None, help="Validation frequency in epochs.")
    parser.add_argument("--exp_name", default=None, help="Explicit experiment name.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for evaluation or resume-like scripts.")
    return parser.parse_args()


def initialize_run(args: argparse.Namespace) -> dict[str, Any]:
    config = apply_overrides(load_config(args.config), args.override)
    if args.wandb_mode is not None:
        config.setdefault("experiment", {})["wandb_mode"] = args.wandb_mode
    if args.eval_during_train:
        config.setdefault("train", {})["eval_during_train"] = True
    if args.eval_freq is not None:
        config.setdefault("train", {})["eval_freq"] = args.eval_freq

    setup_distributed()
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    dirs = prepare_run_dirs(config, exp_name=args.exp_name)
    exp_name = str(dirs["exp_name"])
    logger, log_path = setup_logger(log_root=dirs["log_root"])
    log_config_summary(logger, config)

    if is_main_process():
        save_config(config, Path(dirs["model_dir"]) / "config.yaml")

    wandb_run = init_wandb(
        config=config,
        exp_name=exp_name,
        run_dir=dirs["model_dir"],
        mode=config.get("experiment", {}).get("wandb_mode", "offline"),
        logger=logger,
    )
    return {
        "config": config,
        "dirs": dirs,
        "exp_name": exp_name,
        "logger": logger,
        "log_path": log_path,
        "wandb_run": wandb_run,
    }


def build_cache_manager(config: dict[str, Any], logger: Any) -> CacheManager | None:
    dataset_cfg = config["dataset"]
    paths = config["paths"]
    cache_policy = str(dataset_cfg.get("cache_policy", "validate_or_rebuild")).lower()
    if cache_policy in {"disabled", "none", "off"}:
        logger.info("Cache disabled by dataset.cache_policy=%s", cache_policy)
        return None

    strategy = SamplingStrategy.from_config(dataset_cfg.get("sampling_strategy", {}))
    manager = CacheManager(
        dataset_name=dataset_cfg["name"],
        cache_root=paths["cache_root"],
        sampling_strategy=strategy,
        preprocess_version=str(dataset_cfg.get("preprocess_version", "v1")),
        logger=logger,
    )
    manager.ensure_valid_or_rebuild()
    return manager


def build_dataloader(
    config: dict[str, Any],
    manifest_key: str,
    cache_manager: CacheManager | None,
    logger: Any,
    train: bool,
) -> DataLoader:
    train_cfg = config["train"] if train else config.get("eval", {})
    dataset_cfg = config["dataset"]
    paths = config["paths"]
    skipped_log_path = Path(config.get("experiment", {}).get("log_root", "logs")) / "skipped_samples.log"
    source_format = str(dataset_cfg.get("source_format", "manifest")).lower()

    if source_format == "npz":
        split_name = manifest_key.removeprefix("manifest_")
        npz_path = (
            paths.get(f"{split_name}_npz")
            or paths.get("npz_data")
            or paths.get("data_npz")
        )
        if not npz_path:
            raise KeyError(
                "dataset.source_format=npz requires paths.<split>_npz or paths.npz_data"
            )
        selected_classes = None
        if str(dataset_cfg.get("split", "")).lower() == "zsl":
            if split_name in {"train", "val"}:
                selected_classes = dataset_cfg.get("seen_classes") or None
            elif split_name == "test":
                selected_classes = dataset_cfg.get("unseen_classes") or None

        shape_cfg = dataset_cfg.get("skeleton_shape", {})
        npz_cfg = dataset_cfg.get("npz", {})
        dataset = NpzSkeletonDataset(
            npz_path=npz_path,
            x_key=str(npz_cfg.get("x_key", "x_data")),
            y_key=str(npz_cfg.get("y_key", "y_data")),
            channels=int(shape_cfg.get("channels", 3)),
            joints=int(shape_cfg.get("joints", 25)),
            persons=int(shape_cfg.get("persons", 2)),
            selected_classes=selected_classes,
            skipped_log_path=skipped_log_path,
            logger=logger,
        )
        logger.info(
            "Loaded npz dataset split=%s path=%s samples=%s",
            split_name,
            npz_path,
            len(dataset),
        )
    else:
        dataset = SkeletonDataset(
            manifest_path=paths[manifest_key],
            cache_manager=cache_manager,
            allow_raw_fallback=bool(dataset_cfg.get("allow_raw_fallback", True)),
            skipped_log_path=skipped_log_path,
            logger=logger,
        )
    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=train)
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate,
        drop_last=train,
    )


def load_text_bank(path: str | Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    z_text = payload["z_text"].float().to(device)
    class_ids = torch.arange(z_text.shape[0], dtype=torch.long, device=device)
    return z_text, class_ids


def select_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def move_batch_to_device(batch: dict[str, Any] | None, device: torch.device) -> dict[str, Any] | None:
    if batch is None:
        return None
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def maybe_autocast(enabled: bool, dtype_name: str = "fp16"):
    if not enabled or not torch.cuda.is_available():
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)
