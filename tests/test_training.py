"""Small checks for experiment accounting and fail-closed data requirements."""

import numpy as np
import pytest
import torch

from experiments.sweep_hparams import candidate_grid
from experiments.training import TrainSpec, microbatch_plan, run_training


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
