"""Exact-autodiff Hutchinson samples for Hessian-based baseline optimizers."""

from collections.abc import Callable, Sequence

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from radon.hvp import torch_hvp


def hutchinson_diag_sample(
    model: torch.nn.Module,
    params: Sequence[torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    seed: int,
    absolute: bool = False,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return one signed or absolute Rademacher diagonal-Hessian sample.

    The signed sample is unbiased for the Hessian diagonal. AdaHessian uses
    absolute samples with row averaging for 2D tensors before its moving
    average; Sophia-H uses signed samples.
    """
    vectors = []
    for index, param in enumerate(params):
        generator = torch.Generator(device=param.device)
        generator.manual_seed(seed + index)
        bits = torch.randint(
            0,
            2,
            param.shape,
            generator=generator,
            device=param.device,
            dtype=torch.int8,
        )
        vectors.append(bits.to(param.dtype).mul_(2).sub_(1))

    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        loss = loss_fn(model, inputs, targets) if loss_fn else model(inputs, targets)[1]
        products = torch_hvp(loss, params, vectors)

    samples = []
    for param, vector, product in zip(params, vectors, products):
        diagonal = vector * product
        if absolute:
            diagonal = diagonal.abs()
            if diagonal.ndim == 2:
                diagonal = diagonal.mean(dim=1, keepdim=True).expand_as(diagonal)
        samples.append((param, diagonal.detach()))
    return samples
