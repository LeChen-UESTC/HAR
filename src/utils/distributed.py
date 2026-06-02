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


def wrap_model_for_distributed(model: Any, device: Any | None = None) -> Any:
    if get_world_size() <= 1:
        return model
    import torch
    from torch.nn.parallel import DistributedDataParallel

    if torch.cuda.is_available():
        local_rank = get_local_rank()
        expected_device = torch.device(device) if device is not None else torch.device("cuda", local_rank)
        if expected_device.type != "cuda":
            raise RuntimeError(
                f"Distributed CUDA training requires a CUDA device, got {expected_device}."
            )
        param_devices = {param.device for param in model.parameters()}
        if len(param_devices) != 1 or next(iter(param_devices)) != expected_device:
            raise RuntimeError(
                "DDP requires all model parameters to be on the current local-rank device. "
                f"Expected {expected_device}, got {sorted(str(item) for item in param_devices)}. "
                "For prealign this should be automatic; for Qwen3Embedding4B training do not use "
                "a sharded device_map with DDP."
            )
        return DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    return DistributedDataParallel(model, broadcast_buffers=False)


def reduce_sum(value: float | int, device: Any | None = None) -> float:
    if get_world_size() <= 1:
        return float(value)
    import torch
    import torch.distributed as dist

    if device is None:
        device = torch.device("cuda", get_local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


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
