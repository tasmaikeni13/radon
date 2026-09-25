"""Sophia-H (Second-order Clipped Stochastic Curvature Optimizer) baseline."""

from collections.abc import Callable, Sequence

import torch


class SophiaH(torch.optim.Optimizer):
    """Sophia-H: A Scalable Second-Order Optimizer for Language Model Pre-training.

    Reference: Liu et al., 2023 (arXiv:2305.14342).
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        lr: float = 6e-4,
        betas: tuple[float, float] = (0.96, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
    ):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self, hessian_samples: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Update EMA of diagonal Hessian estimates h."""
        beta2 = self.param_groups[0]["betas"][1]
        for p, h_sample in hessian_samples:
            state = self.state[p]
            if "hessian" not in state:
                state["hessian"] = torch.zeros_like(p)
            state["hessian"].mul_(beta2).add_(h_sample, alpha=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                hess = state["hessian"]

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                denom = (rho * hess).clamp_min(1e-12)
                update = (exp_avg / denom).clamp_(-1.0, 1.0)
                p.add_(update, alpha=-lr)

        return loss
