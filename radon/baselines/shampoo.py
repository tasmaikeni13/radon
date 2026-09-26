"""Blocked matrix Shampoo baseline with per-block Kronecker preconditioners."""

from collections.abc import Callable, Sequence
from itertools import product

import torch

from radon.tpu import mark_step


class Shampoo(torch.optim.Optimizer):
    """Apply matrix Shampoo to 2D parameters and diagonal Adagrad to vectors.

    Matrix parameters are partitioned into blocks so large embedding and MLP
    weights retain Shampoo preconditioning instead of falling back to a
    diagonal method. This class does not implement distributed state sharding.
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 10,
        block_size: int = 128,
    ):
        if lr < 0 or not 0 <= momentum < 1 or epsilon <= 0:
            raise ValueError("Invalid Shampoo learning rate, momentum, or epsilon")
        if update_freq < 1 or block_size < 1:
            raise ValueError("update_freq and block_size must be positive")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
            block_size=block_size,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _inverse_fourth_root(matrix: torch.Tensor, epsilon: float) -> torch.Tensor:
        symmetric = (matrix + matrix.mT) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        return (eigenvectors * eigenvalues.clamp_min(epsilon).pow(-0.25)) @ eigenvectors.mT

    @staticmethod
    def _blocks(rows: int, cols: int, block_size: int) -> list[tuple[int, int, int, int]]:
        return [
            (row, min(row + block_size, rows), col, min(col + block_size, cols))
            for row, col in product(range(0, rows, block_size), range(0, cols, block_size))
        ]

    def _init_state(self, param: torch.Tensor, group: dict) -> dict:
        state = self.state[param]
        if state:
            return state
        state["step"] = 0
        state["momentum"] = torch.zeros_like(param)
        if param.ndim < 2:
            state["sq_grad"] = torch.zeros_like(param, dtype=torch.float32)
            return state
        rows = param.shape[0]
        cols = param.numel() // rows
        slices = self._blocks(rows, cols, group["block_size"])
        state["slices"] = slices
        state["L"] = []
        state["R"] = []
        state["inv_L"] = []
        state["inv_R"] = []
        for row0, row1, col0, col1 in slices:
            row_count = row1 - row0
            col_count = col1 - col0
            eye_l = torch.eye(row_count, device=param.device, dtype=torch.float32)
            eye_r = torch.eye(col_count, device=param.device, dtype=torch.float32)
            state["L"].append(group["epsilon"] * eye_l)
            state["R"].append(group["epsilon"] * eye_r)
            state["inv_L"].append(eye_l)
            state["inv_R"].append(eye_r)
        return state

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            decay = group["weight_decay"]
            epsilon = group["epsilon"]
            update_freq = group["update_freq"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self._init_state(param, group)
                state["step"] += 1
                state["momentum"].mul_(momentum).add_(grad, alpha=1 - momentum)
                if decay:
                    param.mul_(1 - lr * decay)
                if param.ndim < 2:
                    state["sq_grad"].addcmul_(grad.float(), grad.float())
                    denom = state["sq_grad"].sqrt().add_(epsilon)
                    param.add_((state["momentum"].float() / denom).to(param.dtype), alpha=-lr)
                    continue

                rows = param.shape[0]
                cols = param.numel() // rows
                flat_param = param.view(rows, cols)
                flat_grad = grad.reshape(rows, cols)
                flat_momentum = state["momentum"].reshape(rows, cols)
                for index, (row0, row1, col0, col1) in enumerate(state["slices"]):
                    block_grad = flat_grad[row0:row1, col0:col1].float()
                    left = state["L"][index]
                    right = state["R"][index]
                    left.addmm_(block_grad, block_grad.mT)
                    right.addmm_(block_grad.mT, block_grad)
                    if state["step"] % update_freq == 0:
                        state["inv_L"][index] = self._inverse_fourth_root(left, epsilon)
                        state["inv_R"][index] = self._inverse_fourth_root(right, epsilon)
                    block_momentum = flat_momentum[row0:row1, col0:col1].float()
                    update = state["inv_L"][index] @ block_momentum @ state["inv_R"][index]
                    flat_param[row0:row1, col0:col1].add_(update.to(param.dtype), alpha=-lr)
        mark_step()
        return loss
