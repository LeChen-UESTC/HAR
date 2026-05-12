from __future__ import annotations

import os
from typing import Any


def get_rank() -> int:
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def get_world_size() -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> bool:
    world_size = get_world_size()
    if world_size <= 1:
        return False
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return True
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    return True


def cleanup_distributed() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model
