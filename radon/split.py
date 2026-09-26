"""Split curvature channels for RADON (theory.md §3, §5-§6).

The loss Hessian decomposes exactly as H = S + R (Lean: split_exact):
  S = Jᵀ (∇²_logits ℓ) J   — structural core: PSD, known without Hessian products
  R = ℋ(∇ℓ)                — residual: linear in the loss gradient

Two channels estimate the two diagonals:

1. CORE (fisher_diag_sample): for the cross-entropy family the logit Hessian is
   Λ = diag(p) − ppᵀ regardless of targets, so sampled-label gradients yield an
   UNBIASED, ELEMENTWISE-NONNEGATIVE estimate of diag S:
       E[N · g̃ ⊙ g̃] = diag S,   g̃ = ∇θ mean-CE(logits, ŷ),  ŷ ~ softmax(logits).
   Single forward + backward pass; no double-backward required. Each sample is
   non-negative, but finite sampling leaves nonzero structural-channel variance.

2. RESIDUAL (residual_probe): coded sign probes applied to R alone (Lean
   split_probe_unbiased, split_variance_residual_only): R·v = H·v − S·v where
       H·v  is computed via double backward through the loss,
       S·v  = Jᵀ Λ (J v) via the dummy double-VJP trick with closed-form Λ.
   For an exact structural diagonal, residual-probe variance scales with the square
   of the outer-loss gradient (Lean split_variance_scaling). This does not make the
   implemented total estimator noise vanish at a parameter stationary point.
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    HAS_SDPA = True
except ImportError:
    HAS_SDPA = False


def _lift_precision(tensor: torch.Tensor) -> torch.Tensor:
    """Lift half-precision tensors to float32; preserve float32 and float64."""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


def _apply_logit_lambda(probs: torch.Tensor, u: torch.Tensor, num_positions: int) -> torch.Tensor:
    """Apply the mean-CE logit Hessian in closed form: Λu = (p ⊙ u - p (p · u)) / N."""
    inner = (probs * u).sum(dim=-1, keepdim=True)
    return (probs * u - probs * inner) / num_positions


@torch.enable_grad()
def fisher_diag_sample(
    model: torch.nn.Module,
    params: Sequence[torch.Tensor],
    inputs: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate one unbiased, non-negative sample of diag(S).

    Returns a list of (parameter, tensor) pairs in float32.
    """
    was_training = model.training
    model.eval()
    try:
        if HAS_SDPA:
            with sdpa_kernel(SDPBackend.MATH):
                out = model(inputs)
        else:
            out = model(inputs)
        logits = out[0] if isinstance(out, tuple) else out

        vocab_dim = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_dim)
        num_positions = flat_logits.size(0)

        with torch.no_grad():
            probs = F.softmax(_lift_precision(flat_logits), dim=-1)
            sampled_targets = torch.multinomial(probs, 1).squeeze(-1)

        surrogate_loss = F.cross_entropy(flat_logits, sampled_targets)
        grads = torch.autograd.grad(surrogate_loss, params)
    finally:
        if was_training:
            model.train()

    return [(p, (_lift_precision(g) ** 2) * num_positions) for p, g in zip(params, grads)]


@torch.enable_grad()
def residual_probe(
    model: torch.nn.Module,
    params: Sequence[torch.Tensor],
    inputs: torch.Tensor,
    loss_from_logits: Callable[[torch.Tensor], torch.Tensor],
    probes: Sequence[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compute (Rv) ⊙ v for residual R = H - S at the current parameter coordinates.

    Returns a list of (parameter, tensor) pairs in float32.
    """
    was_training = model.training
    model.eval()
    try:
        if HAS_SDPA:
            with sdpa_kernel(SDPBackend.MATH):
                out = model(inputs)
        else:
            out = model(inputs)
        logits = out[0] if isinstance(out, tuple) else out

        num_positions = logits.reshape(-1, logits.size(-1)).size(0)
        loss = loss_from_logits(logits)

        # 1. H·v : double backward through loss
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((gi * vi).sum() for gi, vi in zip(grads, probes))
        hv = torch.autograd.grad(gv, params, retain_graph=True)

        # 2. J·v : dummy double-VJP trick
        dummy = torch.zeros_like(logits, requires_grad=True)
        gtheta = torch.autograd.grad(logits, params, grad_outputs=dummy, create_graph=True, retain_graph=True)
        sv_lin = sum((gi * vi).sum() for gi, vi in zip(gtheta, probes))
        (jv,) = torch.autograd.grad(sv_lin, dummy, retain_graph=True)

        # 3. S·v = Jᵀ Λ (Jv) with closed-form softmax logit Hessian
        with torch.no_grad():
            probs = F.softmax(_lift_precision(logits), dim=-1)
            lambda_jv = _apply_logit_lambda(probs, _lift_precision(jv), num_positions)

        sv = torch.autograd.grad(logits, params, grad_outputs=lambda_jv.to(logits.dtype))

        # 4. (R·v) ⊙ v = (H·v - S·v) ⊙ v
        out = []
        for p, hvi, svi, vi in zip(params, hv, sv, probes):
            rvi = _lift_precision(hvi) - _lift_precision(svi)
            out.append((p, rvi * _lift_precision(vi)))
        return out
    finally:
        if was_training:
            model.train()
