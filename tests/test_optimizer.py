"""Unit tests for RADON optimizer state dict, clamping, and steps."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radon.optimizer import Radon


class TestRadonOptimizer(unittest.TestCase):
    def test_step_and_stats(self):
        w = nn.Parameter(torch.randn(8, 8))
        opt = Radon([w], lr=1e-3, gamma=0.02)
        loss = (w**2).sum()
        loss.backward()

        opt.step()
        stats = opt.stats()
        self.assertIsInstance(stats, dict)

    def test_state_dict_serialization(self):
        w = nn.Parameter(torch.randn(4, 4))
        opt1 = Radon([w], lr=1e-3)
        opt1.total_probes = 16
        opt1.n_commits = 1
        opt1.n_core = 2

        sd = opt1.state_dict()
        opt2 = Radon([w], lr=1e-3)
        opt2.load_state_dict(sd)

        self.assertEqual(opt2.total_probes, 16)
        self.assertEqual(opt2.n_commits, 1)
        self.assertEqual(opt2.n_core, 2)


if __name__ == "__main__":
    unittest.main()
