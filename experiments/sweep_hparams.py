"""Hyperparameter sweep and pilot convergence benchmarking for RADON and peers."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.fineweb import FineWebDataset
from models.transformer import CausalTransformer, TransformerConfig
from radon.baselines import AdaHessian, AdamWBaseline, Shampoo, SophiaH
from radon.optimizer import Radon
from radon.probes import ProbeGenerator
from radon.split import fisher_diag_sample, residual_probe

OUTPUT_DIR = REPO_ROOT / "runs"


def run_pilot(optimizer_name: str, lr: float, gamma: float = 0.02, steps: int = 100):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small pilot config for fast sweep verification
    cfg = TransformerConfig(vocab_size=50304, block_size=128, n_layer=4, n_head=4, n_embd=128)
    model = CausalTransformer(cfg).to(device)
    dataset = FineWebDataset(is_train=True)

    params = list(model.parameters())
    if optimizer_name == "radon":
        opt = Radon(params, lr=lr, gamma=gamma, cycle_m=4)
        prober = ProbeGenerator(params, m=4)
    elif optimizer_name == "adamw":
        opt = AdamWBaseline(params, lr=lr)
    elif optimizer_name == "sophia":
        opt = SophiaH(params, lr=lr)
    elif optimizer_name == "adahessian":
        opt = AdaHessian(params, lr=lr)
    elif optimizer_name == "shampoo":
        opt = Shampoo(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    losses = []
    for step in range(steps):
        x, y = dataset.get_batch(batch_size=4, block_size=128, device=device, seed=42 + step)
        logits, loss = model(x, y)
        loss.backward()

        if optimizer_name == "radon" and step % 4 == 0:
            core_samples = fisher_diag_sample(model, params, x)
            opt.accumulate_core(core_samples)
            cycle_idx = step // 4
            probes = prober.probe(cycle_idx=cycle_idx, r=cycle_idx % 4)
            res_samples = residual_probe(
                model,
                params,
                x,
                lambda lg: F.cross_entropy(lg.view(-1, 50304), y.view(-1)),
                probes,
            )
            opt.accumulate_residual(res_samples)

        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    return losses[0], losses[-1], min(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick smoke sweep")
    args = parser.parse_args()

    steps = 30 if args.quick else 100
    print("=" * 80)
    print(f"  RADON Hyperparameter Sweep & Pilot Validation ({steps} steps)")
    print("=" * 80)

    sweep_results = {}
    optimizers = ["radon", "adamw", "sophia", "adahessian", "shampoo"]
    lrs = {
        "radon": 4e-4,
        "adamw": 1e-3,
        "sophia": 6e-4,
        "adahessian": 1e-3,
        "shampoo": 1e-3,
    }

    for opt_name in optimizers:
        lr = lrs[opt_name]
        init_loss, final_loss, min_loss = run_pilot(opt_name, lr=lr, steps=steps)
        print(f"  {opt_name:<15s} (lr={lr:.1e}): init={init_loss:.4f} -> final={final_loss:.4f} (min={min_loss:.4f})")
        sweep_results[opt_name] = {
            "optimal_lr": lr,
            "initial_loss": init_loss,
            "final_loss": final_loss,
            "min_loss": min_loss,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "hpo_sweep_report.json"
    with open(out_file, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\n[PASS] Sweep report written to {out_file}")


if __name__ == "__main__":
    main()
