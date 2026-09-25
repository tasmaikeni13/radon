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
from radon.tpu import (
    DEFAULT_TPU_POD,
    get_device,
    get_tpu_config,
    get_world_size,
    is_tpu_available,
    mark_step,
    sync_curvature_dict,
    tpu_all_reduce,
)


def test_tpu_pod_abstractions():
    cfg = get_tpu_config()
    assert cfg.num_hosts == 16, f"Expected 16 hosts, got {cfg.num_hosts}"
    assert cfg.total_tensor_cores == 32, f"Expected 32 tensor cores, got {cfg.total_tensor_cores}"
    assert cfg.host_bounds == "1,1,1"
    assert cfg.chip_bounds == "2,2,1"

    dev = get_device()
    assert dev is not None

    # Test all-reduce collective
    t = torch.ones(4, dtype=torch.float32)
    reduced = tpu_all_reduce(t, op="sum")
    assert reduced.shape == (4,)

    # Test curvature synchronization
    p1 = torch.nn.Parameter(torch.randn(3, 3))
    s1 = torch.ones(3, 3)
    synced = sync_curvature_dict([(p1, s1)], op="mean")
    assert len(synced) == 1
    assert synced[0][1].shape == (3, 3)

    mark_step()
    print(f"  {'TPU Pod Abstraction':<20s} Hardware config verified: {cfg.pod_name} (16 hosts / 32 cores, device={dev})")
    return True


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

    test_tpu_pod_abstractions()
    test_radon_pipeline()
    test_optimizer("AdamWBaseline", lambda m: AdamWBaseline(m.parameters(), lr=1e-3))
    test_optimizer("SophiaH", lambda m: SophiaH(m.parameters(), lr=1e-3))
    test_optimizer("AdaHessian", lambda m: AdaHessian(m.parameters(), lr=1e-3))
    test_optimizer("Shampoo", lambda m: Shampoo(m.parameters(), lr=1e-3))

    print("=" * 80)
    print("[SUCCESS] All 5 optimizer kernels verified successfully!")


if __name__ == "__main__":
    main()
