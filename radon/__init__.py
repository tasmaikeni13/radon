"""RADON: residual-aware probing for neural curvature.

A standalone second-order optimizer and curvature estimation framework
built on split-exact curvature theory, coded directional projections,
and native Google Cloud TPU v4-32 Pod slice hardware co-design.
"""

from .adaptive import AdaptiveProbeConfig, adaptive_residual_diagonal, fixed_rademacher_diagonal
from .optimizer import Radon
from .probes import ProbeGenerator, code_tensor, hadamard
from .split import fisher_diag_sample, residual_hvp, residual_probe
from .tpu import (
    DEFAULT_TPU_POD,
    TPUPodConfig,
    get_device,
    get_rank,
    get_tpu_config,
    get_world_size,
    is_master,
    is_tpu_available,
    mark_step,
    sync_curvature_dict,
    tpu_all_reduce,
    tpu_optimizer_step,
    wrap_tpu_loader,
)

__version__ = "1.0.0"
__all__ = [
    "DEFAULT_TPU_POD",
    "AdaptiveProbeConfig",
    "ProbeGenerator",
    "Radon",
    "TPUPodConfig",
    "code_tensor",
    "adaptive_residual_diagonal",
    "fisher_diag_sample",
    "fixed_rademacher_diagonal",
    "get_device",
    "get_rank",
    "get_tpu_config",
    "get_world_size",
    "hadamard",
    "is_master",
    "is_tpu_available",
    "mark_step",
    "residual_probe",
    "residual_hvp",
    "sync_curvature_dict",
    "tpu_all_reduce",
    "tpu_optimizer_step",
    "wrap_tpu_loader",
]
