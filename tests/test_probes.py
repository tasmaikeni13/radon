"""Unit tests for RADON probe machinery (Hadamard codes and Latin coloring)."""

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radon.probes import ProbeGenerator, code_tensor, hadamard


class TestProbes(unittest.TestCase):
    def test_hadamard_orthogonality(self):
        for m in [2, 4, 8, 16, 32]:
            H = hadamard(m)
            self.assertEqual(H.shape, (m, m))
            prod = H @ H.T
            expected = m * torch.eye(m)
            self.assertTrue(torch.allclose(prod, expected, atol=1e-6))

    def test_latin_coloring_row_col(self):
        m = 16
        codes = code_tensor(torch.Size([32, 64]), m=m, device=torch.device("cpu"))
        # Check adjacent row elements have distinct codes
        for r in range(32):
            self.assertNotEqual(codes[r, 0].item(), codes[r, 1].item())
        # Check adjacent col elements have distinct codes
        for c in range(64):
            self.assertNotEqual(codes[0, c].item(), codes[1, c].item())

    def test_probe_generator_cycling(self):
        params = [torch.zeros(16, 32), torch.zeros(16)]
        generator = ProbeGenerator(params, m=8, seed=42)
        probes_0 = generator.probe(cycle_idx=0, r=0)
        probes_1 = generator.probe(cycle_idx=0, r=1)
        self.assertEqual(len(probes_0), 2)
        # Verify elements are +/- 1
        for p in probes_0:
            self.assertTrue(torch.all((p == 1) | (p == -1)))
        # Verify probes change across cycle
        self.assertFalse(torch.allclose(probes_0[0], probes_1[0]))


if __name__ == "__main__":
    unittest.main()
