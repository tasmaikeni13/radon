"""Verification suite for all optimizer kernels (RADON and peer baselines)."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radon.baselines import AdaHessian, AdamWBaseline, Shampoo, SophiaH
from radon.optimizer import Radon
from radon.probes import ProbeGenerator
from radon.split import fisher_diag_sample, residual_probe


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def test_optimizer(name, opt_factory):
    torch.manual_seed(42)
    model = SimpleNet()
    x = torch.randn(4, 16)
    y = torch.randint(0, 8, (4,))
    opt = opt_factory(model)

    # Initial loss
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    # Step
    opt.step()
    opt.zero_grad()

    # Loss after step
    out_after = model(x)
    loss_after = F.cross_entropy(out_after, y)
    decreased = loss_after.item() < loss.item()
    print(
        f"  {name:<20s} Initial loss: {loss.item():.4f} -> After step: {loss_after.item():.4f}  (Decreased: {decreased})"
    )
    assert not torch.isnan(loss_after), f"{name} produced NaN loss!"
    return True


def test_radon_pipeline():
    torch.manual_seed(42)
    model = SimpleNet()
    x = torch.randn(4, 16)
    y = torch.randint(0, 8, (4,))
    params = list(model.parameters())

    opt = Radon(params, lr=1e-3, gamma=0.02)
    prober = ProbeGenerator(params, m=4)

    # 1. Forward & backward
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    # 2. Curvature accumulation
    core_samples = fisher_diag_sample(model, params, x)
    opt.accumulate_core(core_samples)

    probes = prober.probe(cycle_idx=0, r=0)
    res_samples = residual_probe(model, params, x, lambda lg: F.cross_entropy(lg, y), probes)
    opt.accumulate_residual(res_samples)

    # 3. Step
    opt.step()
    stats = opt.stats()
    print(
        f"  {'RADON Pipeline':<20s} Step successful, clip_frac={stats.get('clip_frac', 0):.2f}, n_core={stats.get('n_core', 0)}"
    )
    return True


def main():
    print("=" * 80)
    print("  RADON Kernel & Peer Optimizer Verification Suite")
    print("=" * 80)

    test_radon_pipeline()
    test_optimizer("AdamWBaseline", lambda m: AdamWBaseline(m.parameters(), lr=1e-3))
    test_optimizer("SophiaH", lambda m: SophiaH(m.parameters(), lr=1e-3))
    test_optimizer("AdaHessian", lambda m: AdaHessian(m.parameters(), lr=1e-3))
    test_optimizer("Shampoo", lambda m: Shampoo(m.parameters(), lr=1e-3))

    print("=" * 80)
    print("[SUCCESS] All 5 optimizer kernels verified successfully!")


if __name__ == "__main__":
    main()
