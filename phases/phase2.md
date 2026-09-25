# Phase 2: Statistical & Monte Carlo Analysis: Coded Tomography & Variance Cancellation

## 1. Executive Summary
Phase 2 establishes the statistical superiority of coded tomographic probing over conventional isotropic random sampling (Gaussian, Rademacher, Hutchinson). We analyze cross-talk interference, Latin-square tensor coloring, random sign-flip debiasing, and prove that RADON strictly dominates all peer estimators in variance.

---

## 2. Theoretical Architecture

### Master Probing Identity & Code Coherence
For sign probe $v \in \{-1, +1\}^N$:
$$v_i (M v)_i = M_{ii} + \sum_{j \neq i} M_{ij} v_i v_j$$
Over a cycle of $m$ coded probes $v^1, \dots, v^m$:
$$\frac{1}{m} \sum_{k=1}^m v_i^k (M v^k)_i = M_{ii} + \sum_{j \neq i} M_{ij} \bar{C}_{ij}$$
where $\bar{C}_{ij} = \frac{1}{m} \sum_{k=1}^m v_i^k v_j^k \in [-1, 1]$.

### Latin-Square Code Coloring on Tensor Grids
For 2D parameter matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, we assign Hadamard codes via:
$$\mathrm{code}(a, b) = (a + b) \pmod m$$
- Immediate row neighbors: $\bar{C}_{(a, b_1), (a, b_2)} = 0$.
- Immediate column neighbors: $\bar{C}_{(a_1, b), (a_2, b)} = 0$.
- Dominant off-diagonal cross-talk is eliminated with zero variance.

### Global Random Sign-Flip Debiasing
A single uniform random sign vector $s \sim \{-1, +1\}^N$ per cycle ($v^k \leftarrow s \odot c^k$) ensures:
$$\mathbb{E}[\hat{M}_{ii}] = M_{ii}, \quad \mathrm{Var}(\hat{M}_{ii}) = \sum_{j \neq i} M_{ij}^2 \cdot \bar{C}_{ij}^2$$

---

## 3. Strict Peer Domination Theorems (Lean 4 Verified)
1. `Radon.coded_variance`: Exact closed-form variance of coded probing.
2. `Radon.orthogonal_zero_variance`: Orthogonal coordinate pairs contribute exactly 0 variance.
3. `Radon.coded_le_hutchinson`: $\mathrm{Var}_{\text{RADON}} \le \mathrm{Var}_{\text{Hutchinson}}$ for all coordinates and code choices.
4. `Radon.split_variance_residual_only`: Probing $R$ instead of $H$ yields variance depending only on $R$, while $S$ contributes 0.
5. `Radon.split_variance_scaling`: $\mathrm{Var} \propto \|\nabla \ell\|^2$, contracting to 0 as training converges.

---

## 4. Execution & Verification Gate
```bash
python3 verify/numerical_gate.py --monte-carlo-check
```
**Gate PASS Criteria:**
- Monte Carlo variance of RADON on 2D weight matrix $\le 0.25 \times$ Hutchinson variance.
- Zero variance on orthogonal row/column neighbors within machine precision ($\le 10^{-14}$).
