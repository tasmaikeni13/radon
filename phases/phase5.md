# Phase 5: Numerical Exactness Gate & Microbenchmarks

## 1. Executive Summary
Phase 5 enforces the Numerical Exactness Gate in machine precision (fp64) to confirm that the empirical Python/JAX/PyTorch implementation perfectly reproduces the mathematical theorems verified in Lean 4.

---

## 2. Gate Verification Suite (`verify/numerical_gate.py`)

### 1. Split-Exact Reconstruction Gate
- Compute dense Hessian $H = \nabla^2 \mathcal{L}(\theta)$ via forward-over-reverse autodiff.
- Compute structural core $S = J^\top (\nabla^2 \ell) J$ and residual $R = \mathcal{H}(\nabla \ell)$.
- Verify $\|H - (S + R)\|_F \le 10^{-14}$ (fp64 exactness).

### 2. Zero-Variance Interference Cancellation Gate
- Form 2D parameter weight matrix $W \in \mathbb{R}^{16 \times 16}$.
- Apply Sylvester-Hadamard codes with Latin-square coloring.
- Enumerate all $2^{10}$ sign-flip combinations.
- Confirm variance on orthogonal row/column neighbors is $\le 10^{-14}$.

### 3. Fisher Unbiasedness Gate
- Compute true structural diagonal $\mathrm{diag}(S)$.
- Sample label gradients $\tilde{g}$ over $N_{\text{samples}} = 4000$.
- Confirm empirical expectation $\mathbb{E}[\tilde{g} \odot \tilde{g}]$ converges to $\mathrm{diag}(S)$ within $2\%$ relative error.

### 4. Gradient-Squared Residual Variance Scaling Gate
- Scale loss gradient by $c \in \{0.1, 0.5, 1.0, 2.0\}$.
- Verify that estimator variance scales strictly as $c^2$.

---

## 3. Execution & Verification Gate
```bash
python3 verify/numerical_gate.py
```
**Gate PASS Criteria:**
- All 4 numerical gates return PASS.
- Machine precision errors remain within strict numerical tolerances.
