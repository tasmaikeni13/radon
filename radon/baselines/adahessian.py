"""ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning baseline."""

from collections.abc import Callable, Sequence

import torch

from radon.tpu import mark_step


class AdaHessian(torch.optim.Optimizer):
    """ADAHESSIAN optimizer with Hutchinson diagonal Hessian estimation.

    Reference: Yao et al., 2021 (AAAI-21).
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-4,
        weight_decay: float = 0.0,
        hessian_power: float = 1.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self, hessian_diags: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Update second moment of Hessian diagonal."""
        beta2 = self.param_groups[0]["betas"][1]
        for p, d in hessian_diags:
            state = self.state[p]
            if "hessian" not in state:
                state["hessian"] = torch.zeros_like(p)
            state["hessian"].mul_(beta2).addcmul_(d, d, value=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            power = group["hessian_power"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                hess = state["hessian"]

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step

                h_hat = (hess / bias2).sqrt()
                if power != 1.0:
                    h_hat = h_hat.pow(power)

                denom = h_hat.add(eps)
                step_size = lr / bias1
                update = exp_avg / denom
                p.add_(update, alpha=-step_size)

        mark_step()
        return loss
