"""Shared measured training loop for smoke and owner-run FineWeb experiments."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from data.fineweb import FineWebDataset
from models.transformer import CausalTransformer, TransformerConfig
from radon.baselines import AdaHessian, AdamWBaseline, Shampoo, SophiaH
from radon.baselines.curvature import hutchinson_diag_sample
from radon.hvp import torch_hvp
from radon.optimizer import Radon
from radon.probes import ProbeGenerator, flip_signs
from radon.split import fisher_diag_sample, residual_probe
from radon.tpu import (
    get_device,
    get_rank,
    get_tpu_config,
    get_world_size,
    mark_step,
    tpu_all_reduce,
    tpu_optimizer_step,
)


@dataclass(frozen=True)
class TrainSpec:
    """One optimizer, configuration, and seed under an exact token budget."""

    optimizer: str
    seed: int
    hyperparameters: dict[str, Any]
    steps: int
    tokens_total: int
    block_size: int
    microbatch_size: int
    train_path: Path | None
    valid_path: Path | None
    smoke: bool = False
    eval_batches: int = 8
    variant: str = "standard"


def microbatch_plan(tokens_per_rank_step: int, block_size: int, microbatch_size: int) -> list[tuple[int, int]]:
    """Partition one rank's exact token budget into mostly static shapes."""
    if min(tokens_per_rank_step, block_size, microbatch_size) < 1:
        raise ValueError("Token, context, and microbatch sizes must be positive")
    remaining = tokens_per_rank_step
    plan = []
    while remaining >= block_size:
        batch = min(microbatch_size, remaining // block_size)
        plan.append((batch, block_size))
        remaining -= batch * block_size
    if remaining:
        plan.append((1, remaining))
    assert sum(batch * length for batch, length in plan) == tokens_per_rank_step
    return plan


def cosine_lr(step: int, steps: int, base_lr: float) -> float:
    """Warm up, then decay to ten percent of the base learning rate."""
    warmup = min(2000, max(1, steps // 5))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    fraction = (step - warmup) / max(1, steps - warmup - 1)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction)))


def build_model(spec: TrainSpec, device: torch.device) -> CausalTransformer:
    if spec.smoke:
        cfg = TransformerConfig(
            vocab_size=50304, block_size=spec.block_size,
            n_layer=4, n_head=4, n_embd=128,
        )
    else:
        cfg = TransformerConfig(
            vocab_size=50304, block_size=spec.block_size,
            n_layer=12, n_head=12, n_embd=768,
        )
    return CausalTransformer(cfg).to(device)


def build_optimizer(
    spec: TrainSpec, params: list[torch.Tensor]
) -> tuple[torch.optim.Optimizer, ProbeGenerator | None]:
    hp = spec.hyperparameters
    name = spec.optimizer
    if name == "radon":
        cycle_m = int(hp.get("cycle_m", 16))
        optimizer = Radon(
            params,
            lr=float(hp["lr"]),
            gamma=float(hp.get("gamma", 0.02)),
            cycle_m=cycle_m,
            betas=tuple(hp.get("betas", (0.96, 0.95))),
            beta_core=float(hp.get("beta_core", 0.95)),
            weight_decay=float(hp.get("weight_decay", 0.1)),
        )
        return optimizer, ProbeGenerator(params, m=cycle_m, seed=spec.seed)
    if name == "adamw":
        return AdamWBaseline(
            params, lr=float(hp["lr"]),
            betas=tuple(hp.get("betas", (0.9, 0.95))),
            weight_decay=float(hp.get("weight_decay", 0.1)),
        ), None
    if name == "sophia":
        return SophiaH(
            params, lr=float(hp["lr"]),
            rho=float(hp.get("rho", 0.04)),
            betas=tuple(hp.get("betas", (0.96, 0.99))),
            weight_decay=float(hp.get("weight_decay", 0.1)),
        ), None
    if name == "adahessian":
        return AdaHessian(
            params, lr=float(hp["lr"]),
            hessian_power=float(hp.get("hessian_power", 1.0)),
            betas=tuple(hp.get("betas", (0.9, 0.999))),
            weight_decay=float(hp.get("weight_decay", 0.0)),
        ), None
    if name == "shampoo":
        return Shampoo(
            params, lr=float(hp["lr"]),
            update_freq=int(hp.get("update_freq", 10)),
            block_size=int(hp.get("block_size", 128)),
            weight_decay=float(hp.get("weight_decay", 0.0)),
        ), None
    raise ValueError(f"Unknown optimizer: {name}")


def _update_curvature(
    spec: TrainSpec,
    optimizer: torch.optim.Optimizer,
    prober: ProbeGenerator | None,
    model: CausalTransformer,
    params: list[torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    step: int,
) -> None:
    if step % int(spec.hyperparameters.get("probe_freq", 4)):
        return
    if spec.optimizer == "radon":
        assert isinstance(optimizer, Radon) and prober is not None
        frequency = int(spec.hyperparameters.get("probe_freq", 4))
        probe_index = step // frequency
        if spec.variant != "full_hessian":
            optimizer.accumulate_core(fisher_diag_sample(model, params, inputs))
        if spec.variant == "isotropic":
            vectors = [
                flip_signs(p.shape, spec.seed + 1_000_003 * probe_index + i, p.device, p.dtype)
                for i, p in enumerate(params)
            ]
        else:
            vectors = prober.scheduled_probe(step, frequency)
        if spec.variant == "full_hessian":
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                _, probe_loss = model(inputs, targets)
                hv = torch_hvp(probe_loss, params, vectors)
            samples = [(p, v * product) for p, v, product in zip(params, vectors, hv)]
        else:
            samples = residual_probe(
                model, params, inputs,
                lambda logits: F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                ),
                vectors,
            )
        optimizer.accumulate_residual(samples)
    elif spec.optimizer in {"sophia", "adahessian"}:
        samples = hutchinson_diag_sample(
            model, params, inputs, targets,
            seed=spec.seed + step,
            absolute=spec.optimizer == "adahessian",
        )
        optimizer.update_hessian(samples)


@torch.no_grad()
def evaluate(
    model: CausalTransformer,
    dataset: FineWebDataset,
    spec: TrainSpec,
    device: torch.device,
    rank: int,
) -> tuple[float, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    for index in range(spec.eval_batches):
        inputs, targets = dataset.get_batch(
            batch_size=spec.microbatch_size,
            block_size=spec.block_size,
            device=device,
            seed=spec.seed + 10_000_000 + rank * spec.eval_batches + index,
        )
        _, loss = model(inputs, targets)
        count = targets.numel()
        loss_sum += loss.double() * count
        token_count += count
    loss_sum = tpu_all_reduce(loss_sum, op="sum")
    token_count = tpu_all_reduce(token_count, op="sum")
    val_loss = (loss_sum / token_count).item()
    model.train()
    return val_loss, math.exp(val_loss)


def run_training(spec: TrainSpec, device: torch.device | None = None) -> dict[str, Any]:
    """Train one seed and return only metrics measured by this invocation."""
    if device is None:
        device = get_device()
    if spec.steps < 1 or spec.eval_batches < 1:
        raise ValueError("steps and eval_batches must be positive")
    if spec.variant not in {"standard", "isotropic", "full_hessian"}:
        raise ValueError(f"Unknown ablation variant: {spec.variant}")
    if spec.variant != "standard" and spec.optimizer != "radon":
        raise ValueError("Curvature variants require RADON")
    world_size = get_world_size()
    rank = get_rank()
    denominator = world_size * spec.steps
    if spec.tokens_total % denominator:
        raise ValueError("tokens_total must divide exactly across ranks and steps")
    tokens_per_rank_step = spec.tokens_total // denominator
    plan = microbatch_plan(tokens_per_rank_step, spec.block_size, spec.microbatch_size)

    if not spec.smoke and (spec.train_path is None or spec.valid_path is None):
        raise ValueError("Full training requires explicit train and validation paths")
    train_data = FineWebDataset(data_path=spec.train_path, is_train=True)
    valid_data = FineWebDataset(data_path=spec.valid_path, is_train=False)
    if not spec.smoke:
        if train_data.synthetic or valid_data.synthetic:
            raise FileNotFoundError("Full training requires real train and validation token arrays")
        if spec.train_path.resolve() == spec.valid_path.resolve():
            raise ValueError("Train and validation token arrays must be distinct files")
        if train_data.num_tokens < spec.tokens_total:
            raise ValueError("Training array is smaller than the requested token budget")
        if valid_data.num_tokens <= spec.block_size:
            raise ValueError("Validation array is shorter than one context plus target")
        if device.type == "xla" and world_size != 16:
            raise RuntimeError("TPU v4-32 full runs require 16 workers across the pod")
    torch.manual_seed(spec.seed)
    model = build_model(spec, device)
    params = list(model.parameters())
    optimizer, prober = build_optimizer(spec, params)
    last_train_loss = float("nan")
    processed_tokens = 0
    started = time.perf_counter()
    for step in range(spec.steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, spec.steps, float(spec.hyperparameters["lr"]))
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        curvature_batch = None
        for micro_index, (batch_size, context) in enumerate(plan):
            data_seed = (
                spec.seed * 1_000_003 + step * len(plan) * world_size
                + micro_index * world_size + rank
            )
            inputs, targets = train_data.get_batch(
                batch_size=batch_size,
                block_size=context,
                device=device,
                seed=data_seed,
            )
            processed_tokens += targets.numel() * world_size
            if curvature_batch is None:
                curvature_batch = (inputs, targets)
            _, loss = model(inputs, targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            weight = batch_size * context / tokens_per_rank_step
            (loss * weight).backward()
            weighted_loss += loss.detach().item() * weight
        assert curvature_batch is not None
        _update_curvature(spec, optimizer, prober, model, params, *curvature_batch, step)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm).item():
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        tpu_optimizer_step(optimizer)
        mark_step()
        last_train_loss = weighted_loss

    if processed_tokens != spec.tokens_total:
        raise RuntimeError(f"Processed {processed_tokens} tokens, expected {spec.tokens_total}")
    elapsed = tpu_all_reduce(
        torch.tensor(time.perf_counter() - started, device=device, dtype=torch.float64),
        op="max",
    ).item()
    val_loss, val_ppl = evaluate(model, valid_data, spec, device, rank)
    hardware = (
        get_tpu_config().pod_name if device.type == "xla"
        else torch.cuda.get_device_name(device) if device.type == "cuda"
        else "CPU"
    )
    return {
        "optimizer": spec.optimizer,
        "variant": spec.variant,
        "seed": spec.seed,
        "hyperparameters": spec.hyperparameters,
        "steps": spec.steps,
        "tokens_total": processed_tokens,
        "tokens_per_rank_step": tokens_per_rank_step,
        "microbatch_plan": plan,
        "model_params": sum(param.numel() for param in params),
        "model_config": asdict(model.cfg),
        "block_size": spec.block_size,
        "hardware": hardware,
        "device_type": device.type,
        "world_size": world_size,
        "train_file": str(spec.train_path),
        "valid_file": str(spec.valid_path),
        "synthetic_train": train_data.synthetic,
        "synthetic_valid": valid_data.synthetic,
        "final_train_loss": last_train_loss,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "elapsed_seconds": elapsed,
        "mean_step_time_ms": elapsed * 1000 / spec.steps,
        "smoke": spec.smoke,
    }
