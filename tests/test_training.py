"""Small checks for experiment accounting and fail-closed data requirements."""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import training as training_module
from experiments.sweep_hparams import candidate_grid
from experiments.training import TrainSpec, microbatch_plan, run_training, validate_measured_result
from models.transformer import CausalTransformer, TransformerConfig
from radon.tpu import sync_curvature_dict, tpu_optimizer_step


def _distributed_curvature_worker(rank, store_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank,
        world_size=2, timeout=timedelta(seconds=30),
    )
    try:
        param = torch.nn.Parameter(torch.zeros(2))
        sample = torch.full((2,), float(rank + 1))
        pooled = sync_curvature_dict([(param, sample)], op="mean")
        assert torch.allclose(pooled[0][1], torch.full((2,), 1.5))
        optimizer = torch.optim.SGD([param], lr=1.0)
        param.grad = torch.full((2,), float(rank + 1))
        tpu_optimizer_step(optimizer)
        assert torch.allclose(param.detach(), torch.full((2,), -1.5))
    finally:
        dist.destroy_process_group()


def test_exact_token_plan_and_registered_search_space():
    assert microbatch_plan(350, block_size=128, microbatch_size=2) == [
        (2, 128), (1, 94)
    ]
    assert microbatch_plan(15_625, block_size=2048, microbatch_size=2) == [
        (2, 2048), (2, 2048), (2, 2048), (1, 2048), (1, 1289)
    ]
    assert {name: len(candidates) for name, candidates in candidate_grid().items()} == {
        "radon": 36, "adamw": 12, "sophia": 9, "adahessian": 6, "shampoo": 12
    }


def test_full_training_requires_real_disjoint_arrays(tmp_path):
    missing = TrainSpec(
        optimizer="adamw", seed=42, hyperparameters={"lr": 1e-3},
        steps=1, tokens_total=350, block_size=128, microbatch_size=2,
        train_path=tmp_path / "missing_train.npy",
        valid_path=tmp_path / "missing_valid.npy",
    )
    with pytest.raises(FileNotFoundError, match="real train and validation"):
        run_training(missing, torch.device("cpu"))

    token_file = tmp_path / "tokens.npy"
    np.save(token_file, np.arange(500, dtype=np.int32))
    same_split = TrainSpec(
        optimizer="adamw", seed=42, hyperparameters={"lr": 1e-3},
        steps=1, tokens_total=350, block_size=128, microbatch_size=2,
        train_path=token_file, valid_path=token_file,
    )
    with pytest.raises(ValueError, match="distinct files"):
        run_training(same_split, torch.device("cpu"))


def test_curvature_samples_are_pooled_across_workers(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_curvature_worker,
        args=(str(tmp_path / "gloo_store"),),
        nprocs=2,
        join=True,
    )


def test_real_data_training_path_with_tiny_model(tmp_path, monkeypatch):
    train_file = tmp_path / "train.npy"
    valid_file = tmp_path / "valid.npy"
    np.save(train_file, np.arange(500, dtype=np.int32) % 64)
    np.save(valid_file, (np.arange(500, dtype=np.int32) + 7) % 64)

    def tiny_model(spec, device):
        config = TransformerConfig(
            vocab_size=64, block_size=spec.block_size,
            n_layer=1, n_head=2, n_embd=32,
        )
        return CausalTransformer(config).to(device)

    monkeypatch.setattr(training_module, "build_model", tiny_model)
    spec = TrainSpec(
        optimizer="adamw", seed=42, hyperparameters={"lr": 1e-3},
        steps=1, tokens_total=350, block_size=128, microbatch_size=2,
        train_path=train_file, valid_path=valid_file, eval_batches=1,
    )
    result = run_training(spec, torch.device("cpu"))
    assert result["tokens_total"] == 350
    assert result["synthetic_train"] is False
    assert result["synthetic_valid"] is False
    assert result["model_config"]["vocab_size"] == 64
    assert result["hardware"] == "CPU"
    assert result["hvp_calls_per_rank"] == 0
    assert np.isfinite(result["val_loss"])
    validate_measured_result(result, {
        "steps": 1, "tokens_total": 350,
        "train_file": str(train_file), "valid_file": str(valid_file),
    })
    tampered = dict(result, val_ppl=result["val_ppl"] * 2)
    with pytest.raises(ValueError, match="perplexity differs"):
        validate_measured_result(tampered, {"tokens_total": 350})
    tampered_count = dict(result, hvp_calls_per_rank=1)
    with pytest.raises(ValueError, match="HVP call count"):
        validate_measured_result(tampered_count, {"tokens_total": 350})
