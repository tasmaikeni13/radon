"""Hessian-vector product (HVP) and operator sweep routines for RADON.

Supports both PyTorch autodiff (for GPU/CPU execution) and JAX/XLA
forward-over-reverse automatic differentiation (for TPU Pod slices).
"""

from typing import Any, Callable, List, Sequence

import torch

try:
    import jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def torch_hvp(
    loss: torch.Tensor, params: Sequence[torch.Tensor], vectors: Sequence[torch.Tensor]
) -> List[torch.Tensor]:
    """Compute Hessian-vector product Hv via PyTorch reverse-over-reverse autodiff."""
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vectors))
    hv = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    return list(hv)


if HAS_JAX:

    def jax_forward_over_reverse_hvp(f: Callable, primals: Any, tangents: Any) -> Any:
        """Exact forward-over-reverse Hessian-vector product on TPU TensorCores."""
        return jax.jvp(jax.grad(f), (primals,), (tangents,))[1]
