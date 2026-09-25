"""Distributed Shampoo (Block-Kronecker Curvature Optimizer) baseline."""

from collections.abc import Callable, Sequence

import torch
from radon.tpu import mark_step


class Shampoo(torch.optim.Optimizer):
    """Shampoo: Preconditioned Stochastic Tensor Optimization.

    Reference: Gupta et al., 2018 (ICML).
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 10,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            eps = group["epsilon"]
            update_freq = group["update_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)
                    if p.ndim >= 2 and max(p.shape) <= 1024:
                        state["L"] = eps * torch.eye(p.shape[0], device=p.device, dtype=p.dtype)
                        state["R"] = eps * torch.eye(p.shape[1], device=p.device, dtype=p.dtype)
                        state["inv_L"] = torch.eye(p.shape[0], device=p.device, dtype=p.dtype)
                        state["inv_R"] = torch.eye(p.shape[1], device=p.device, dtype=p.dtype)
                    else:
                        state["sq_grad"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                m = state["momentum"]

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                m.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                if "L" in state:
                    flat_g = grad.reshape(p.shape[0], -1)
                    state["L"].addmm_(flat_g, flat_g.t())
                    state["R"].addmm_(flat_g.t(), flat_g)

                    if step % update_freq == 0:
                        try:
                            # SVD matrix fourth root inverse
                            u_l, s_l, _ = torch.linalg.svd(state["L"])
                            state["inv_L"] = u_l @ torch.diag(s_l.clamp_min(eps).pow(-0.25)) @ u_l.t()
                            u_r, s_r, _ = torch.linalg.svd(state["R"])
                            state["inv_R"] = u_r @ torch.diag(s_r.clamp_min(eps).pow(-0.25)) @ u_r.t()
                        except Exception:
                            pass

                    precond = state["inv_L"] @ m.reshape(p.shape[0], -1) @ state["inv_R"]
                    p.add_(precond.reshape(p.shape), alpha=-lr)
                else:
                    sq = state["sq_grad"]
                    sq.addcmul_(grad, grad)
                    denom = sq.sqrt().add_(eps)
                    p.addcdiv_(m, denom, value=-lr)

        mark_step()
        return loss
