"""Verification suite for all optimizer kernels (RADON and peer baselines)."""

import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radon.adaptive import AdaptiveProbeConfig, adaptive_residual_diagonal
from radon.baselines import AdaHessian, AdamWBaseline, Shampoo, SophiaH
from radon.baselines.curvature import hutchinson_diag_sample
from radon.optimizer import Radon
from radon.probes import ProbeGenerator
from radon.split import fisher_diag_sample, residual_hvp, residual_probe
from radon.tpu import (
    get_device,
    get_tpu_config,
    mark_step,
    sync_curvature_dict,
    tpu_all_reduce,
)


def test_tpu_pod_abstractions():
    cfg = get_tpu_config()
    assert cfg.num_hosts == 4, f"Expected 4 hosts, got {cfg.num_hosts}"
    assert cfg.total_chips == cfg.num_hosts * cfg.chips_per_host
    assert cfg.total_tensor_cores == 32, f"Expected 32 tensor cores, got {cfg.total_tensor_cores}"
    assert cfg.host_bounds == "1,1,4"
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
    print(f"  {'TPU Pod Abstraction':<20s} Configuration checked: {cfg.pod_name} (4 hosts / 16 chips / 32 cores, device={dev})")
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

    if name in {"SophiaH", "AdaHessian"}:
        samples = hutchinson_diag_sample(
            model, list(model.parameters()), x, y, seed=42,
            absolute=name == "AdaHessian",
            loss_fn=lambda network, inputs, targets: F.cross_entropy(network(inputs), targets),
        )
        opt.update_hessian(samples)
        assert any(sample.abs().sum().item() > 0 for _, sample in samples)

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

    opt = Radon(params, lr=1e-3, gamma=0.02, cycle_m=4)
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


def test_adaptive_radon_pipeline():
    """Exercise learned directions and variable probe count through exact autodiff."""
    torch.manual_seed(43)
    model = SimpleNet()
    inputs = torch.randn(4, 16)
    targets = torch.randint(0, 8, (4,))
    params = list(model.parameters())
    optimizer = Radon(params, lr=1e-3, gamma=0.02, cycle_m=1)
    F.cross_entropy(model(inputs), targets).backward()
    optimizer.accumulate_core(fisher_diag_sample(model, params, inputs))
    def matvec(vectors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return residual_hvp(
            model, params, inputs,
            lambda logits: F.cross_entropy(logits, targets), vectors,
        )
    estimate = adaptive_residual_diagonal(
        matvec, params,
        AdaptiveProbeConfig(target_relative_rms=0.25, max_final_probes=2),
        seed=43,
    )
    assert 4 <= estimate.hvp_calls <= 6
    assert all(torch.isfinite(item).all() for item in estimate.diagonal)
    optimizer.accumulate_residual(list(zip(params, estimate.diagonal)))
    optimizer.step()
    assert optimizer.n_commits == 1
    print(f"  {'Adaptive RADON':<20s} Exact residual HVP calls: {estimate.hvp_calls}")
    return True


def main():
    print("=" * 80)
    print("  RADON Kernel & Peer Optimizer Verification Suite")
    print("=" * 80)

    test_tpu_pod_abstractions()
    test_radon_pipeline()
    test_adaptive_radon_pipeline()
    test_optimizer("AdamWBaseline", lambda m: AdamWBaseline(m.parameters(), lr=1e-3))
    test_optimizer("SophiaH", lambda m: SophiaH(m.parameters(), lr=1e-3))
    test_optimizer("AdaHessian", lambda m: AdaHessian(m.parameters(), lr=1e-3))
    test_optimizer("Shampoo", lambda m: Shampoo(m.parameters(), lr=1e-3, update_freq=1, block_size=16))

    print("=" * 80)
    print("[SUCCESS] All 5 optimizer kernels verified successfully!")


if __name__ == "__main__":
    main()
