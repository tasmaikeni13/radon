# Phase 1: Mathematical & Formal Analysis of Curvature Carriers and Split-Exact Decomposition

## 1. Executive Summary
Phase 1 formalizes curvature-carrier algebra and the split-exact Hessian identity. The structural core is positive semidefinite when the outer-loss Hessian is positive semidefinite. The residual is linear in the outer-loss gradient and vanishes if that gradient is zero. Lean checks the stated algebraic identities with no project-specific axioms or `sorry`; it does not formalize a neural training run.

---

## 2. Theoretical Architecture

### Carrier Algebra
For a smooth map $f : \mathbb{R}^n \to \mathbb{R}^m$, curvature is carried by the pair:
$$\mathcal{C}(f; x) = (J, \mathcal{H})$$
where $J = Df(x) \in \mathbb{R}^{m \times n}$ and $\mathcal{H}(\lambda) = \nabla^2 \langle \lambda, f \rangle(x) \in \mathrm{Sym}(n)$.

### Pullback Composition Law
For composite map $h = g \circ f$:
$$J_h = J_g J_f, \quad \mathcal{H}_h(\lambda) = J_f^\top \mathcal{H}_g(\lambda) J_f + \mathcal{H}_f(J_g^\top \lambda)$$

### The Split-Exact Decomposition
For loss $\mathcal{L} = \ell \circ f$:
$$H = S + R$$
- $S = J_f^\top (\nabla^2 \ell) J_f \succeq 0$ when $\nabla^2\ell\succeq 0$ (Structural Core)
- $R = \mathcal{H}_f(\nabla \ell)$ (Residual, linear in $\nabla \ell$)

---

## 3. Formal Lean 4 Verification Targets
The following theorems must compile with zero `sorry` in `proofs/RadonCert/RadonCert/Radon.lean`:
1. `Radon.comp_assoc`: Associativity of carrier composition.
2. `Radon.split_exact`: Exactness of $H = S + R$.
3. `Radon.core_posCore`: Positive semidefiniteness of structural core $S$.
4. `Radon.posCore_diag_nonneg`: Non-negativity of $\mathrm{diag}(S)$.
5. `Radon.residual_smul`: Linearity of residual in loss gradient.
6. `Radon.residual_zero`: Vanishing residual when the outer-loss gradient is zero. A zero parameter gradient alone is insufficient.

---

## 4. Execution & Verification Gate
```bash
cd proofs/RadonCert && lake build
```
**Gate PASS Criteria:**
- Exit code 0.
- Zero `sorry` statements.
- Axiom check: only `{propext, Classical.choice, Quot.sound}`.
