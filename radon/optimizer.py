"""RADON optimizer — clipped preconditioned momentum on a split-exact curvature diagonal.

Curvature state incorporates two complementary channels reflecting H = S + R:
* s : EMA of unbiased non-negative structural core samples (sampled-label Fisher diagonals).
      Guaranteed non-negative, preventing preconditioner sign-flips.
* r : EMA over completed residual-probe cycles of coded products v ⊙ (Rv). Over a cycle of
      m Hadamard-Latin coded probes, row and column cross-talk cancels exactly.

Total curvature: ĥ = ŝ + r̂ estimates diag(H). The structural sample and residual
probe both contribute estimation noise; only the ideal residual-probe variance is
governed exclusively by R.

Update: θ ← θ − lr · clamp( μ̂ / max(γ · ĥ, ε), −1, 1 ) − lr · λ · θ
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from .tpu import get_world_size, mark_step, tpu_all_reduce


class Radon(torch.optim.Optimizer):
    """RADON: Residual-aware Antithetic Decoupled Orthogonal Newton Optimizer."""

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        lr: float = 4e-4,
        betas: tuple[float, float] = (0.96, 0.95),
        beta_core: float = 0.95,
        gamma: float = 0.02,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        cycle_m: int = 16,
        r_weight: float = 1.0,
        sync_across_tpu: bool = True,
    ):
        if eps < 1e-12:
            raise ValueError("eps must be at least 1e-12")
        if cycle_m < 1 or cycle_m & (cycle_m - 1):
            raise ValueError("cycle_m must be a positive power of two")
        if lr < 0 or gamma <= 0 or r_weight < 0:
            raise ValueError("lr and r_weight must be nonnegative and gamma must be positive")
        if not 0 <= beta_core < 1 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas and beta_core must be in [0, 1)")
        defaults = dict(
            lr=lr,
            betas=betas,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            cycle_m=cycle_m,
        )
        super().__init__(params, defaults)
        self.cycle_m = cycle_m
        self.beta_core = beta_core
        self.r_weight = r_weight
        self.sync_across_tpu = sync_across_tpu
        self.total_probes = 0
        self.n_commits = 0
        self.n_core = 0

    def _state(self, p: torch.Tensor) -> dict[str, Any]:
        st = self.state[p]
        if len(st) == 0:
            st["mu"] = torch.zeros_like(p)
            st["s"] = torch.zeros_like(p)
            st["r"] = torch.zeros_like(p)
            st["racc"] = torch.zeros_like(p)
            st["t"] = 0
        return st

    @torch.no_grad()
    def accumulate_core(self, core_samples: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Commit one non-negative core sample per parameter into the s-EMA."""
        if self.sync_across_tpu and get_world_size() > 1:
            core_samples = [(p, tpu_all_reduce(cs.clone(), op="mean")) for p, cs in core_samples]
        for p, cs in core_samples:
            st = self._state(p)
            st["s"].mul_(self.beta_core).add_(cs, alpha=1 - self.beta_core)
        self.n_core += 1

    @torch.no_grad()
    def accumulate_residual(self, probe_products: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Add one residual probe product v ⊙ (Rv) per parameter; commit at cycle completion."""
        if self.sync_across_tpu and get_world_size() > 1:
            probe_products = [(p, tpu_all_reduce(pp.clone(), op="mean")) for p, pp in probe_products]
        for p, pp in probe_products:
            st = self._state(p)
            st["racc"].add_(pp)
        self.total_probes += 1
        if self.total_probes % self.cycle_m == 0:
            beta2 = self.param_groups[0]["betas"][1]
            for group in self.param_groups:
                for p in group["params"]:
                    st = self._state(p)
                    st["r"].mul_(beta2).add_(st["racc"], alpha=(1 - beta2) / self.cycle_m)
                    st["racc"].zero_()
            self.n_commits += 1

    def _channel_scales(self) -> tuple[float, bool, float]:
        """Compute bias-correction scaling factors (sscale, use_acc, rscale)."""
        beta2 = self.param_groups[0]["betas"][1]
        sscale = 0.0 if self.n_core == 0 else 1.0 / (1.0 - self.beta_core**self.n_core)
        w = self.r_weight
        if self.n_commits == 0:
            rpart = self.total_probes % self.cycle_m
            if rpart == 0:
                return sscale, False, 0.0
            return sscale, True, w / rpart
        return sscale, False, w / (1.0 - beta2**self.n_commits)

    def _h_hat(self, st: dict[str, Any]) -> torch.Tensor | None:
        """Compute bias-corrected fused curvature diagonal ĥ = ŝ + r̂."""
        sscale, use_acc, rscale = self._channel_scales()
        if sscale == 0.0 and rscale == 0.0:
            return None
        rsrc = st["racc"] if use_acc else st["r"]
        return st["s"] * sscale + rsrc * rscale

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None if closure is None else closure()
        sscale, use_acc, rscale = self._channel_scales()
        no_info = sscale == 0.0 and rscale == 0.0

        for group in self.param_groups:
            lr = group["lr"]
            beta1, _ = group["betas"]
            gamma, eps, wd = group["gamma"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st["t"] += 1
                bias1 = 1.0 - beta1 ** st["t"]

                st["mu"].mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
                mu_hat = st["mu"] / bias1

                if no_info:
                    denom = torch.full_like(p, eps)
                else:
                    h_hat = self._h_hat(st)
                    denom = (gamma * h_hat).clamp_min(eps)

                u = (mu_hat / denom).clamp_(-1.0, 1.0)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr)

        mark_step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["radon_extra"] = {
            "total_probes": self.total_probes,
            "n_commits": self.n_commits,
            "n_core": self.n_core,
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        extra = state_dict.pop("radon_extra", None)
        super().load_state_dict(state_dict)
        if extra is not None:
            self.total_probes = extra["total_probes"]
            self.n_commits = extra["n_commits"]
            self.n_core = extra["n_core"]

    @torch.no_grad()
    def stats(self) -> dict[str, float]:
        """Compute summary statistics for curvature and trust-region clipping."""
        hs, ss, rs, clip_n, tot = [], [], [], 0, 0
        for group in self.param_groups:
            beta1, _ = group["betas"]
            gamma, eps = group["gamma"], group["eps"]
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                h_hat = self._h_hat(st)
                if h_hat is None:
                    continue
                sscale, use_acc, rscale = self._channel_scales()
                hs.append(h_hat.flatten())
                ss.append((st["s"] * sscale).flatten())
                rs.append(((st["racc"] if use_acc else st["r"]) * rscale).flatten())
                mu_hat = st["mu"] / (1.0 - beta1 ** max(st["t"], 1))
                u = mu_hat / (gamma * h_hat).clamp_min(eps)
                clip_n += (u.abs() >= 1.0).sum().item()
                tot += u.numel()

        if not hs:
            return {}

        h = torch.cat(hs).float()
        s = torch.cat(ss).float()
        r = torch.cat(rs).float()
        sample_size = min(200_000, h.numel())
        idx = torch.randint(0, h.numel(), (sample_size,), device=h.device)
        quantiles = torch.quantile(h[idx], torch.tensor([0.05, 0.5, 0.95], device=h.device))
        return {
            "h_p05": quantiles[0].item(),
            "h_p50": quantiles[1].item(),
            "h_p95": quantiles[2].item(),
            "h_neg_frac": (h < 0).float().mean().item(),
            "r_share": (r.abs().sum() / (s.sum() + r.abs().sum() + 1e-30)).item(),
            "clip_frac": clip_n / max(tot, 1),
            "n_commits": self.n_commits,
            "n_core": self.n_core,
        }
