/-
RADON adaptive projection: fixed-state identities.

The choice of q may depend on any pilot observations. Once q is fixed, fresh
uniform sign probes estimate the remaining diagonal without bias. The theorems
here concern a fixed matrix and a fresh sign cube. They do not assert that the
calibration-based stopping rule reaches a requested accuracy or that a Hessian
remains fixed while neural parameters are updated.
-/
import RadonCert.Radon

open Finset Matrix

noncomputable section

namespace Radon

variable {n : ℕ}

/-- The matrix left after removing the rank-one operator `(M q) qᵀ`. -/
def adaptiveRemainder (M : Matrix (Fin n) (Fin n) ℝ) (q : Fin n → ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  fun i j => M i j - M.mulVec q i * q j

/-- One fresh sign probe plus the exact diagonal of the removed rank-one part. -/
def adaptiveProbe (M : Matrix (Fin n) (Fin n) ℝ) (q : Fin n → ℝ)
    (s : Fin n → Bool) (i : Fin n) : ℝ :=
  M.mulVec q i * q i + sgn s i * (adaptiveRemainder M q).mulVec (sgn s) i

/-- Projection changes the direction and variance but preserves the exact diagonal split. -/
theorem adaptive_diag_split (M : Matrix (Fin n) (Fin n) ℝ)
    (q : Fin n → ℝ) (i : Fin n) :
    M.mulVec q i * q i + adaptiveRemainder M q i i = M i i := by
  simp [adaptiveRemainder]

/-- Conditional unbiasedness: for any pilot-selected `q`, fresh uniform signs
average to the original diagonal. The all-zero `q` is the fixed random baseline. -/
theorem adaptive_probe_unbiased (M : Matrix (Fin n) (Fin n) ℝ)
    (q : Fin n → ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool, adaptiveProbe M q s i = (2 : ℝ) ^ n * M i i := by
  unfold adaptiveProbe
  rw [Finset.sum_add_distrib]
  have hconst : (∑ _s : Fin n → Bool, M.mulVec q i * q i) =
      (2 : ℝ) ^ n * (M.mulVec q i * q i) := by
    simp [Finset.sum_const]
  rw [hconst, hutchinson_unbiased]
  rw [← mul_add]
  exact congrArg ((2 : ℝ) ^ n * ·) (adaptive_diag_split M q i)

/-- Exact finite-cube second moment for any pilot-selected direction. -/
theorem adaptive_probe_variance (M : Matrix (Fin n) (Fin n) ℝ)
    (q : Fin n → ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool, (adaptiveProbe M q s i - M i i) ^ 2
      = (2 : ℝ) ^ n *
          ∑ j ∈ univ.erase i, (adaptiveRemainder M q i j) ^ 2 := by
  have hpoint : ∀ s : Fin n → Bool,
      adaptiveProbe M q s i - M i i =
        sgn s i * (adaptiveRemainder M q).mulVec (sgn s) i -
          adaptiveRemainder M q i i := by
    intro s
    unfold adaptiveProbe
    have h := adaptive_diag_split M q i
    linarith
  rw [Finset.sum_congr rfl fun s _ => by rw [hpoint s]]
  exact hutchinson_variance (adaptiveRemainder M q) i

/-- A data-dependent pilot choice is allowed because the variance theorem is
uniform in the selected direction; the final sign cube remains fresh. -/
theorem adaptive_selected_variance {Pilot : Type*}
    (select : Pilot → Fin n → ℝ) (pilot : Pilot)
    (M : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool,
        (adaptiveProbe M (select pilot) s i - M i i) ^ 2
      = (2 : ℝ) ^ n *
          ∑ j ∈ univ.erase i, (adaptiveRemainder M (select pilot) i j) ^ 2 :=
  adaptive_probe_variance M (select pilot) i

/-- An exact structural-core diagonal can be added without changing the
conditional unbiasedness of adaptive residual probing. -/
theorem adaptive_core_fusion_unbiased
    (S R : Matrix (Fin n) (Fin n) ℝ) (q : Fin n → ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool, (S i i + adaptiveProbe R q s i)
      = (2 : ℝ) ^ n * (S + R) i i := by
  rw [Finset.sum_add_distrib]
  have hconst : (∑ _s : Fin n → Bool, S i i) = (2 : ℝ) ^ n * S i i := by
    simp [Finset.sum_const]
  rw [hconst, adaptive_probe_unbiased]
  simp [Matrix.add_apply, mul_add]

/-- With an exact core diagonal, all adaptive probe variance comes from the
residual remainder, even when its direction was selected from pilot data. -/
theorem adaptive_core_fusion_variance
    (S R : Matrix (Fin n) (Fin n) ℝ) (q : Fin n → ℝ) (i : Fin n) :
    ∑ s : Fin n → Bool,
        (S i i + adaptiveProbe R q s i - (S + R) i i) ^ 2
      = (2 : ℝ) ^ n *
          ∑ j ∈ univ.erase i, (adaptiveRemainder R q i j) ^ 2 := by
  have hpoint : ∀ s : Fin n → Bool,
      S i i + adaptiveProbe R q s i - (S + R) i i
        = adaptiveProbe R q s i - R i i := by
    intro s
    simp [Matrix.add_apply]
  rw [Finset.sum_congr rfl fun s _ => by rw [hpoint s]]
  exact adaptive_probe_variance R q i

end Radon
