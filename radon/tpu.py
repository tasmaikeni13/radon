"""Hardware Pod & Acceleration Abstraction for Google Cloud TPU v4-32 Pod Slices.

Target Hardware Architecture:
- Google Cloud TPU v4-32 Pod Slice
- 4 TPU v4 host nodes
- 32 TPU v4 TensorCore engines (2 TensorCores per v4 chip)
- 32 GiB HBM2e per chip (1.2 TB/s memory bandwidth per chip)
- Interconnect topology: 2x2x4 chip mesh
- Host bounds: TPU_HOST_BOUNDS="1,1,4"
- Chip bounds: TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"

Provides unified device detection, collective communication (all-reduce, broadcast),
XLA computation graph lowering (mark_step), and distributed loader wrapping with
seamless fallback to CUDA or CPU.
"""

import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch

# Check if PyTorch/XLA is available
_HAS_XLA = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    _HAS_XLA = True
except ImportError:
    _HAS_XLA = False


@dataclass(frozen=True)
class TPUPodConfig:
    """Hardware configuration specification for Google Cloud TPU v4-32 Pod Slice."""
    pod_name: str = "Google Cloud TPU v4-32 Pod Slice"
    num_hosts: int = 4
    chips_per_host: int = 4
    tensor_cores_per_chip: int = 2
    total_chips: int = 16
    total_tensor_cores: int = 32
    hbm_per_chip_gb: float = 32.0
    total_hbm_gb: float = 512.0
    peak_tflops_per_core_bf16: float = 137.5
    total_peak_tflops_bf16: float = 4400.0
    host_bounds: str = "1,1,4"
    chip_bounds: str = "2,2,1"


DEFAULT_TPU_POD = TPUPodConfig()


def get_tpu_config() -> TPUPodConfig:
    """Returns the TPU Pod hardware configuration."""
    return DEFAULT_TPU_POD


def is_tpu_available() -> bool:
    """Returns True if running on an authentic Google Cloud TPU device with torch_xla."""
    if not _HAS_XLA:
        return False
    try:
        dev = xm.xla_device()
        return xm.xla_device_hw(dev) == "TPU"
    except Exception:
        return False


def get_device() -> torch.device:
    """Returns the optimal compute device (XLA TPU -> CUDA GPU -> CPU fallback)."""
    # Check for forced CPU fallback via environment
    if os.environ.get("JAX_PLATFORMS", "").lower() == "cpu":
        return torch.device("cpu")

    if is_tpu_available():
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_world_size() -> int:
    """Returns the total number of distributed worker devices across the TPU pod."""
    if is_tpu_available():
        try:
            return xm.xrt_world_size()
        except Exception:
            pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    """Returns global ordinal rank of current worker across the TPU pod."""
    if is_tpu_available():
        try:
            return xm.get_ordinal()
        except Exception:
            pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_local_rank() -> int:
    """Returns local ordinal rank on the current host node."""
    if is_tpu_available():
        try:
            return xm.get_local_ordinal()
        except Exception:
            pass
    return 0


def is_master() -> bool:
    """Returns True if this worker is the global rank 0 (master) ordinal."""
    return get_rank() == 0


def mark_step() -> None:
    """Triggers XLA computation graph lowering and lazy-tensor compilation on TPU.

    Safe no-op when executing on CPU or CUDA.
    """
    if is_tpu_available():
        xm.mark_step()


def tpu_all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """Synchronizes a tensor across all TPU v4-32 cores via high-speed ICI interconnect.

    Args:
        tensor: PyTorch tensor to reduce across devices.
        op: Reduction operation ('sum', 'mean', 'min', 'max').

    Returns:
        Synchronized reduced tensor.
    """
    if is_tpu_available():
        reduce_type = xm.REDUCE_SUM if op.lower() in ("sum", "mean") else op.lower()
        reduced = xm.all_reduce(reduce_type, tensor)
        if op.lower() == "mean":
            reduced = reduced / get_world_size()
        return reduced

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist_op = torch.distributed.ReduceOp.SUM
        if op.lower() == "min":
            dist_op = torch.distributed.ReduceOp.MIN
        elif op.lower() == "max":
            dist_op = torch.distributed.ReduceOp.MAX
        torch.distributed.all_reduce(tensor, op=dist_op)
        if op.lower() == "mean":
            tensor = tensor / torch.distributed.get_world_size()
        return tensor

    return tensor


def tpu_optimizer_step(optimizer: torch.optim.Optimizer, barrier: bool = True) -> None:
    """Executes optimizer parameter step on TPU Pod with gradient sync and graph lowering.

    Calls xm.optimizer_step on XLA TPU devices; calls standard optimizer.step() on CPU/CUDA.
    """
    if is_tpu_available():
        xm.optimizer_step(optimizer, barrier=barrier)
        return
    optimizer.step()


def wrap_tpu_loader(loader: Any, device: torch.device) -> Any:
    """Wraps PyTorch DataLoader with MpDeviceLoader for asynchronous TPU HBM streaming."""
    if is_tpu_available() and "xla" in str(device).lower():
        try:
            return pl.MpDeviceLoader(loader, device)
        except Exception:
            pass
    return loader


def sync_curvature_dict(
    curvature_samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    op: str = "mean",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Synchronizes structural Fisher core or residual probe products across all TPU cores.

    Ensures that in multi-core distributed TPU v4-32 training, curvature estimates
    are pooled across the entire cluster batch B_cluster = 32 * B_local.
    """
    if get_world_size() <= 1:
        return list(curvature_samples)

    synced = []
    for p, sample in curvature_samples:
        reduced_sample = tpu_all_reduce(sample.clone(), op=op)
        synced.append((p, reduced_sample))
    return synced
