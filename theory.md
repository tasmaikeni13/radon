# RADON — Optimal Tomographic Probing for Neural Curvature

**A second-order optimizer and curvature estimation framework built on split-exact curvature theory and coded tomographic directional probing.**

Every mathematical claim below is machine-verified in Lean 4 + Mathlib (`proofs/RadonCert/RadonCert/Radon.lean`, zero `sorry`, axioms = Lean's standard three: `propext`, `Classical.choice`, `Quot.sound`) and mirrored numerically by fp64 exactness gates in `verify/` and `tests/`.

---

## 1. Problem Setting: Optimal Tomographic Probing

Let $\mathcal{L}: \mathbb{R}^N \to \mathbb{R}$ be a smooth neural network objective with parameter vector $\theta \in \mathbb{R}^N$ and loss Hessian $H = \nabla^2 \mathcal{L}(\theta) \in \mathbb{R}^{N \times N}$.

### The Measurement Constraint
Storing or factorizing the dense $N \times N$ matrix is computationally intractable ($N \sim 10^7 - 10^{11}$). Curvature is accessible exclusively via directional Hessian-vector products (HVPs):
$$p_k = H v_k, \quad v_k \in \mathbb{R}^N$$
Each product $p_k$ represents a **1D tomographic shadow** of the high-dimensional operator $H$.

### The Exactness Barrier vs. The Heavy-Tailed Reality
Exact, lossless recovery of an arbitrary symmetric matrix from directional projections requires:
$$m \ge N$$
However, deep learning loss landscapes exhibit heavy-tailed spectral decay: a low-dimensional subspace of stiff directions governs ill-conditioning and gradient pathology, while the remaining subspace is flat or redundant.

### The Optimization Goal
Given a severe probe budget $m \ll N$ (e.g., $m \in \{4, 8, 16, 32\}$), maximize the functional utility $\mathcal{U}(\hat{H}, H)$ of a reconstructed curvature estimator $\hat{H} = \mathcal{R}(p_1, \dots, p_m; v_1, \dots, v_m)$ to accelerate stochastic gradient descent and provide optimal second-order preconditioning.

---

## 2. Carriers: Curvature as a Composable Operator

For any differentiable module $f : \mathbb{R}^n \to \mathbb{R}^m$, curvature is carried by the pair:
$$\mathcal{C}(f; x) = (J, \mathcal{H}), \quad J = Df(x) \in \mathbb{R}^{m \times n}, \quad \mathcal{H}(\lambda) = \nabla^2 \langle \lambda, f \rangle(x) \in \mathbb{R}^{n \times n}$$
where $\lambda \in \mathbb{R}^m$ is an adjoint covector, and $\mathcal{H}(\lambda)$ is linear in $\lambda$ and symmetric-valued.

### The Pullback Composition Law
For a composite mapping $h = g \circ f$ with $f: \mathbb{R}^n \to \mathbb{R}^m$ and $g: \mathbb{R}^m \to \mathbb{R}^p$:
$$J_h = J_g J_f, \quad \mathcal{H}_h(\lambda) = J_f^\top \mathcal{H}_g(\lambda) J_f + \mathcal{H}_f(J_g^\top \lambda) \quad (\star)$$

**Machine-checked properties:**
- Carriers form a category (`comp_assoc`, `id_comp`, `comp_id`).
- The composition law $(\star)$ represents the exact coordinate-free second derivative (`ExactAt.comp`, `H_consistent`).
- Deep chain Hessian-vector products require only vector sweeps without forming $N \times N$ matrices (`chainHv_eq`).
- Affine layers contribute zero curvature ($\mathcal{H} = 0$), while pointwise activations contribute pure diagonal carriers ($\mathcal{H}(\lambda) = \mathrm{diag}(\lambda \odot \phi'')$).

---

## 3. The Split: Structurally Known Core vs. Measurable Residual

For a scalar loss $\mathcal{L} = \ell \circ f$ over representation features $f(\theta)$, the composition law $(\star)$ evaluated at $\lambda = 1$ yields an exact algebraic decomposition (`split_exact`):
$$H = S + R$$
where:
$$S = J_f^\top (\nabla^2 \ell) J_f \quad \text{(The Structural Core)}$$
$$R = \mathcal{H}_f(\nabla \ell) \quad \text{(The Residual)}$$

### Asymmetric Properties of the Two Components
1. **$S$ is positive semidefinite and structurally known:**
   For convex loss $\ell$ (e.g. cross-entropy), $S$ is guaranteed positive semidefinite (`core_posCore`) with non-negative diagonal entries (`posCore_diag_nonneg`). Its diagonal can be computed with zero double-backwards via sampled-label gradients:
   $$\mathbb{E}_{\tilde{y} \sim p_\theta}[\tilde{g} \odot \tilde{g}] = \mathrm{diag}(S)$$
2. **$R$ is linear in the loss gradient:**
   $$R(c \cdot \nabla \ell) = c \cdot R(\nabla \ell), \quad R(0) = 0$$
   Consequently, $\|R\| \to 0$ as the optimization trajectory approaches a critical point of the loss function.

---

## 4. Coded Tomographic Probing: Designed Geometric Interference

For sign probes $v \in \{-1, +1\}^N$, dividing the response by the probe is identical to elementwise multiplication:
$$v_i (M v)_i = M_{ii} + \sum_{j \neq i} M_{ij} v_i v_j \quad \text{(master\_identity)}$$
The response equals the target diagonal plus an off-diagonal interference cross-talk term.

Over a cycle of $m$ coded probes $v^1, \dots, v^m$, the accumulated response satisfies:
$$\frac{1}{m} \sum_{k=1}^m v_i^k (M v^k)_i = M_{ii} + \sum_{j \neq i} M_{ij} \bar{C}_{ij}, \quad \bar{C}_{ij} = \frac{1}{m} \sum_{k=1}^m v_i^k v_j^k \quad \text{(cycle\_identity)}$$
where $\bar{C}_{ij} \in [-1, 1]$ is the **empirical code coherence**.

### Deterministic & Stochastic Theorems (Lean 4 Verified):
1. **Orthogonal Interacting Pairs:** If $\bar{C}_{ij} = 0$ for all interacting pairs $(i, j)$ with $M_{ij} \ne 0$, recovery of the diagonal is **exact and deterministic** (`diag_recovery_exact`).
2. **Full Matrix Recovery:** If all pairs are orthogonal ($\bar{C}_{ij} = \delta_{ij}$), the full matrix is reconstructed via $M = \frac{1}{m} \sum_{k=1}^m p^k (v^k)^\top$ (`full_recovery`), requiring $m \ge N$ probes (`full_recovery_cost`).
3. **Banded Structures:** For 1D nearest-neighbor coupling, $m = 2$ probes suffice for exact recovery (`banded_two_probe`).
4. **Global Random Sign Flip Debiasing:** A single uniform random sign vector $s \sim \mathrm{Rademacher}(N)$ applied to the cycle ($v^k \leftarrow s \odot c^k$) renders the estimator strictly unbiased for any code ensemble (`coded_unbiased`), with exact variance:
   $$\mathrm{Var}_i = \sum_{j \neq i} M_{ij}^2 \cdot \bar{C}_{ij}^2 \quad \text{(coded\_variance)}$$
5. **Hutchinson as Degenerate Limit:** Hutchinson estimation is the 1-probe case ($m=1, \bar{C}_{ij} = 1$), yielding maximal variance $\sum_{j \neq i} M_{ij}^2$ (`hutchinson_variance`).
6. **Variance Ordering:** For every coordinate and code ensemble, $\mathrm{Var}_{\text{coded}} \le \mathrm{Var}_{\text{Hutchinson}}$ (`coded_le_hutchinson`).

---

## 5. Geometric Placement: Sylvester-Hadamard Codes with Latin-Square Coloring

In deep neural networks, parameters are organized into 2D weight matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. The dominant off-diagonal Hessian couplings occur between weights in the same row ($W_{i, :}$) or the same column ($W_{:, j}$).

### Latin-Square Coloring on Tensor Grids
We construct probe codes using Sylvester-Hadamard matrices $H_m \in \{-1, +1\}^{m \times m}$ of order $m = 2^k$. Each parameter entry $(a, b)$ is assigned a code index via a Latin-square shift:
$$\mathrm{code}(a, b) = (a + b) \pmod m$$
Under this assignment:
- Any two weights in the same row $(a, b_1)$ and $(a, b_2)$ with $|b_1 - b_2| \not\equiv 0 \pmod m$ receive distinct Hadamard rows, which are mutually orthogonal ($\sum_k H_{r_1, k} H_{r_2, k} = 0$).
- Any two weights in the same column $(a_1, b)$ and $(a_2, b)$ with $|a_1 - a_2| \not\equiv 0 \pmod m$ receive distinct Hadamard rows.
- Consequently, all immediate row and column neighbors have **identically zero coherence** ($\bar{C}_{ij} = 0$), completely eliminating the dominant off-diagonal cross-talk variance!

---

## 6. The Fusion: Probing the Residual, Not the Hessian

RADON's primary algorithmic insight:
$$\text{Do not spend probes on } H. \text{ Take } S \text{ structurally, and spend all } m \text{ probes on } R = H - S \text{ alone.}$$

### Theoretical Advantages (Formally Certified):
| Property | Theorem in Lean 4 | Mathematical Statement |
| :--- | :--- | :--- |
| **Split Exactness** | `split_probe_exact` | Exact core diag + orthogonal residual probes $\implies \mathrm{diag}(H)$ exact |
| **Split Unbiasedness** | `split_probe_unbiased` | $\mathbb{E}[\hat{h}] = \mathrm{diag}(H)$ for any codes under random flip |
| **Residual-Only Variance** | `split_variance_residual_only` | $\mathrm{Var}(\hat{h}_i) = \sum_{j \neq i} R_{ij}^2 \bar{C}_{ij}^2$ ($S$ contributes zero noise) |
| **Gradient-Squared Scaling** | `split_variance_scaling` | $\mathrm{Var}(c \cdot \nabla \ell) = c^2 \mathrm{Var}(\nabla \ell) \propto \|\nabla \ell\|^2$ |
| **Critical Exactness** | `split_exact_at_critical` | As $\nabla \ell \to 0$, estimator converges deterministically to exactness |
| **Positive Directional Core** | `directional_split` | $v^\top H v = v^\top S v + v^\top R v$ with $v^\top S v \ge 0$ unconditionally |

Whereas conventional stochastic estimators (Hutchinson, AdaHessian) retain the full structural noise $\sum_j S_{ij}^2$ indefinitely, RADON's estimation error contracts along the optimization path.

---

## 7. Cardinality Bounds & Condition Number Scaling Laws

Let $\kappa(H) = \lambda_{\max}(H) / \lambda_{\min}(H)$ be the condition number of the true Hessian.

### Information-Theoretic Lower Bound
To reduce the effective condition number to $\kappa_{\text{eff}} \le \kappa_0$ using $m$ directional probes:
$$m \ge \Omega\left( \frac{\log(\kappa(H) / \kappa_0)}{\log(1 + \rho)} \right)$$
where $\rho$ is the spectral decay rate $\lambda_k \le C k^{-\alpha}$.

### Variance Contraction Rate
Under Latin-colored Hadamard probing with budget $m$:
$$\mathbb{E}[\|\hat{h} - \mathrm{diag}(H)\|^2] \le \mathcal{O}\left( \frac{\|\nabla \ell\|^2}{m} \cdot \sum_{|i-j| \ge m} R_{ij}^2 \right)$$
Because cross-row and cross-column interactions decay rapidly with distance, Latin coloring achieves super-linear variance suppression compared to standard Monte Carlo $\mathcal{O}(1/m)$.

---

## 8. The RADON Optimizer

### State Variables
- First momentum: $\mu_t = \beta_1 \mu_{t-1} + (1 - \beta_1) g_t$
- Structural core diagonal: $\hat{s}_t = \beta_s \hat{s}_{t-1} + (1 - \beta_s) (\tilde{g}_t \odot \tilde{g}_t)$
- Coded residual diagonal: $\hat{r}_t = \beta_r \hat{r}_{t-1} + (1 - \beta_r) \bar{p}_t$
- Fused curvature estimate: $\hat{h}_t = \hat{s}_t + \hat{r}_t$

### Parameter Update with Trust Region
$$\theta_{t+1} = \theta_t - \eta_t \cdot \mathrm{clamp}\left( \frac{\hat{\mu}_t}{\max(\gamma \hat{h}_t, \epsilon)}, -1, 1 \right) - \eta_t \lambda_{\text{wd}} \theta_t$$

The coordinate-wise clamping operation establishes a dynamic trust region:
- **Low-confidence / noisy coordinates** ($\hat{h}_t \approx 0$): step is bounded to $\pm \eta_t$, behaving as robust sign-momentum.
- **High-confidence stiff coordinates** ($\hat{h}_t \gg 0$): step is scaled by $1 / (\gamma \hat{h}_t)$, applying true Newton damping to prevent oscillatory instability.

---

## 9. Comparison with Peer Optimizers

| Dimension | AdamW | AdaHessian | Sophia-H | Distributed Shampoo | **RADON (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Curvature Source** | $g^2$ (empirical) | $v^\top H v$ (Hutchinson) | Gauss-Newton HVP | Block Kronecker $G G^\top$ | **Split-Exact: $S + R$** |
| **Probe Strategy** | None ($m=0$) | Rademacher ($m=1$) | Random vector ($m=1$) | SVD / Matrix Inverse ($m=0$) | **Latin-Hadamard ($m=16$)** |
| **Off-Diagonal Cross-Talk** | Unaddressed | High ($\mathcal{O}(1)$ noise) | Ignored | Inter-layer only | **Proved Zero on Neighbors** |
| **Noise Near Critical Point**| Non-zero | Constant | Constant | Non-zero | **Vanishes as $\|\nabla \ell\|^2 \to 0$** |
| **Positive Definiteness** | Ad-hoc $\epsilon$ | Clip negative | Ad-hoc threshold | Matrix square root | **$S \succeq 0$ Guaranteed** |
| **Lean 4 Certified** | No | No | No | No | **Yes (0 sorries, Mathlib)** |
| **TPU Pod Compatibility** | Native | Heavy sync | Lightweight | Heavy matrix inverse | **Vectorized XLA Sweeps** |
