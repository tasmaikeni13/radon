"""Numerical Exactness Gate for RADON — High-Precision Mirror of Lean 4 Theorems.

Executes machine-precision (fp64) checks against dense autograd ground truth:
  G1  split_exact               : H == S + R entrywise (S from dense J and closed-form Λ,
                                    R from independent weighted-logit Hessian)
  G2  residual_products         : residual_probe == (H - S)v ⊙ v
  G3  core_posCore              : dense S is PSD; diag(S) >= 0
  G4  full_recovery             : (1/m) Σ p^k (v^k)ᵀ == R with all-pairs orthogonal codes
  G5  coded_variance            : exact enumeration of all 2^10 flips matches Σ_{j≠i} M_ij² C̄_ij²
  G6  variance_scaling          : scaling gradient by c scales residual variance by c²
  G7  split_probe_exact         : diag(S) + coded diag-recovery of R == diag(H) exactly
  G8  fisher_unbiasedness       : sampled-label Fisher diagonal converges to diag(S)
  G9  one_step_newton           : exact diagonal minimizes diagonal quadratic in 1 step
  G10 zero_orthogonal_variance  : orthogonal row/column pairs contribute exactly 0 variance

Exit code 0 iff every gate passes.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from radon.probes import hadamard
from radon.split import fisher_diag_sample, residual_probe

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

FAILS = []
RESULTS = {}


def check(name: str, err: float, tol: float) -> bool:
    ok = err < tol
    status = "PASS" if ok else "FAIL"
    print(f"  {name:<44s} err={err:.3e}  tol={tol:.0e}  {status}")
    RESULTS[name] = {"error": float(err), "tolerance": float(tol), "passed": bool(ok)}
    if not ok:
        FAILS.append(name)
    return ok


class ToyModel(torch.nn.Module):
    """2-layer tanh net, 5-way CE head (~117 parameters)."""

    def __init__(self, d_in: int = 4, d_h: int = 7, n_cls: int = 5):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_in, d_h, bias=False)
        self.fc2 = torch.nn.Linear(d_h, n_cls, bias=False)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        logits = self.fc2(torch.tanh(self.fc1(x)))
        if targets is None:
            return logits
        return F.cross_entropy(logits, targets)


def flatten_tensors(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def main():
    print("=" * 80)
    print("  RADON Numerical Exactness Gate (fp64 Machine-Precision Verification)")
    print("=" * 80)

    model = ToyModel()
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    batch_size = 3
    x = torch.randn(batch_size, 4)
    y = torch.randint(0, 5, (batch_size,))

    def loss_fn(lg):
        return F.cross_entropy(lg, y)

    theta0 = flatten_tensors(params).clone()

    def loss_of_theta(theta):
        outs, off = [], 0
        for p in params:
            outs.append(theta[off : off + p.numel()].view_as(p))
            off += p.numel()
        act = x @ outs[0].T
        lg = torch.tanh(act) @ outs[1].T
        return F.cross_entropy(lg, y)

    # 1. Dense ground-truth Hessian
    h_dense = torch.autograd.functional.hessian(loss_of_theta, theta0)
    h_dense = 0.5 * (h_dense + h_dense.T)

    # 2. Dense structural core S = Jᵀ Λ J / B
    def logits_of_theta(theta):
        outs, off = [], 0
        for p in params:
            outs.append(theta[off : off + p.numel()].view_as(p))
            off += p.numel()
        return (torch.tanh(x @ outs[0].T) @ outs[1].T).reshape(-1)

    j_dense = torch.autograd.functional.jacobian(logits_of_theta, theta0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).reshape(-1, 5)

    blocks = []
    for p in probs:
        lam = torch.diag(p) - torch.outer(p, p)
        blocks.append(lam)
    lam_full = torch.block_diag(*blocks) / (batch_size * 1.0)
    s_dense = j_dense.T @ lam_full @ j_dense
    s_dense = 0.5 * (s_dense + s_dense.T)
    # Construct R independently as Σ_k (∂ℓ/∂logit_k) ∇² logit_k.
    logits_for_grad = logits_of_theta(theta0.detach().requires_grad_(True))
    logit_grad = torch.autograd.grad(
        F.cross_entropy(logits_for_grad.reshape(batch_size, 5), y),
        logits_for_grad,
    )[0].detach()
    r_dense = torch.autograd.functional.hessian(
        lambda theta: (logits_of_theta(theta) * logit_grad).sum(),
        theta0,
    )
    r_dense = 0.5 * (r_dense + r_dense.T)

    # G1: Split-Exact Decomposition
    check(
        "G1 split_exact (H == S + R)",
        (h_dense - (s_dense + r_dense)).abs().max().item(),
        1e-14,
    )

    # G2: Residual Probing Exactness
    v_probe = [torch.randn_like(p) for p in params]
    v_flat = flatten_tensors(v_probe)
    res_probe_out = residual_probe(model, params, x, loss_fn, v_probe)
    res_flat = flatten_tensors([t for _, t in res_probe_out])
    expected_res = (r_dense @ v_flat) * v_flat
    check(
        "G2 residual_probe == (H-S)v ⊙ v",
        (res_flat - expected_res).abs().max().item(),
        1e-12,
    )

    # G3: Core Positive Semidefiniteness
    eigvals = torch.linalg.eigvalsh(s_dense)
    min_eig = eigvals.min().item()
    check("G3 core_posCore (S is PSD)", max(0.0, -min_eig), 1e-12)
    min_diag = torch.diag(s_dense).min().item()
    check("G3 posCore_diag_nonneg (diag(S) >= 0)", max(0.0, -min_diag), 1e-14)

    # G4: Full Matrix Recovery via Orthogonal Codes
    m_had = 1
    while m_had < n:
        m_had *= 2
    Hm = hadamard(m_had).to(r_dense.dtype)
    V = Hm[:, :n]  # (m, n) column codes: distinct columns are mutually orthogonal
    reconstructed_r = torch.zeros(n, n)
    for k in range(m_had):
        vk = V[k]
        reconstructed_r += torch.outer(r_dense @ vk, vk)
    reconstructed_r /= float(m_had)
    check(
        "G4 full_matrix_recovery (m >= n)",
        (reconstructed_r - r_dense).abs().max().item(),
        1e-10,
    )

    # G5: Exact Enumeration of Coded Variance on 10x10 Submatrix
    sub_m = r_dense[:10, :10]
    m_code = 4
    had4 = hadamard(m_code).to(sub_m.dtype)
    codes = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    code_matrix = had4[:, codes]  # (4, 10)
    c_bar = (code_matrix.T @ code_matrix) / float(m_code)

    flips = []
    for i in range(1024):
        bits = [(i >> b) & 1 for b in range(10)]
        flips.append([1.0 if b == 1 else -1.0 for b in bits])
    all_flips = torch.tensor(flips, dtype=sub_m.dtype)

    diag_m = torch.diag(sub_m)
    estimates = []
    for s in all_flips:
        cycle_est = torch.zeros(10)
        for r in range(m_code):
            vr = s * had4[r, codes]
            p = sub_m @ vr
            cycle_est += p * vr
        estimates.append(cycle_est / float(m_code))
    all_est = torch.stack(estimates)
    empirical_var = torch.mean((all_est - diag_m) ** 2, dim=0)

    theoretical_var = torch.zeros(10)
    for i in range(10):
        for j in range(10):
            if i != j:
                theoretical_var[i] += (sub_m[i, j] ** 2) * (c_bar[i, j] ** 2)

    check(
        "G5 coded_variance (all 2^10 flips)",
        (empirical_var - theoretical_var).abs().max().item(),
        1e-12,
    )

    # G6: Variance Gradient-Squared Scaling
    c_factor = 2.5
    scaled_r = torch.autograd.functional.hessian(
        lambda theta: (logits_of_theta(theta) * (c_factor * logit_grad)).sum(),
        theta0,
    )
    check(
        "G6 residual scales with outer gradient",
        (scaled_r - c_factor * r_dense).abs().max().item(),
        1e-12,
    )
    scaled_var = torch.zeros(10)
    for i in range(10):
        for j in range(10):
            if i != j:
                scaled_var[i] += (scaled_r[i, j] ** 2) * (c_bar[i, j] ** 2)
    check(
        "G6 variance_scaling (scales as c^2)",
        (scaled_var - (c_factor**2) * theoretical_var).abs().max().item(),
        1e-12,
    )

    # G7: Split Estimator Diagonal Recovery
    split_diag_est = torch.diag(s_dense).clone()
    coded_r_diag = torch.zeros(n)
    for k in range(m_had):
        vk = V[k]
        coded_r_diag += vk * (r_dense @ vk)
    split_diag_est += coded_r_diag / float(m_had)
    check(
        "G7 split_probe_exact (diag(S) + R_probe == diag(H))",
        (split_diag_est - torch.diag(h_dense)).abs().max().item(),
        1e-10,
    )

    # G8: Fisher Diagonal Unbiasedness
    torch.manual_seed(2)
    n_fisher_samples = 4000
    accum_fisher = torch.zeros(n)
    for _ in range(n_fisher_samples):
        samples = fisher_diag_sample(model, params, x)
        accum_fisher += flatten_tensors([t for _, t in samples])
    mean_fisher = accum_fisher / float(n_fisher_samples)
    rel_fisher_err = ((mean_fisher - torch.diag(s_dense)).abs().sum() / torch.diag(s_dense).abs().sum()).item()
    check("G8 fisher_unbiasedness (rel L1 < 8%)", rel_fisher_err, 0.08)

    # G9: One-Step Newton Minimization
    h_pos = torch.abs(torch.diag(h_dense)) + 1.0
    x_quad = torch.randn(n)
    b_quad = torch.randn(n)
    # Loss: 0.5 x^T H x + b^T x -> grad = H x + b. Newton step: x - grad / H = - b / H
    grad_quad = h_pos * x_quad + b_quad
    x_next = x_quad - grad_quad / h_pos
    grad_after = h_pos * x_next + b_quad
    check(
        "G9 one_step_newton (grad after step == 0)",
        grad_after.abs().max().item(),
        1e-14,
    )

    # G10: Zero Variance on Orthogonal Pairs
    orthogonal_matrix = torch.zeros_like(sub_m)
    orthogonal_matrix[0, 1] = orthogonal_matrix[1, 0] = 1.0
    ortho_estimates = []
    for s in all_flips:
        cycle_est = torch.zeros(10)
        for r in range(m_code):
            vr = s * had4[r, codes]
            cycle_est += (orthogonal_matrix @ vr) * vr
        ortho_estimates.append(cycle_est / m_code)
    ortho_error = torch.stack(ortho_estimates).abs().max().item()
    check("G10 zero_orthogonal_variance (tested probe cycle)", ortho_error, 1e-14)

    print("=" * 80)
    if FAILS:
        print(f"[FAIL] {len(FAILS)} gates failed: {FAILS}")
        sys.exit(1)
    else:
        print("[SUCCESS] All 10 Numerical Exactness Gates PASSED cleanly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
