# Phase 3: Hardware Acceleration & High-Performance Kernel Suite for TPU v4 Pods

## 1. Executive Summary
Phase 3 implements hardware-accelerated kernels for Google Cloud TPU v4 Pod slices (specifically 16 TPU v4 accelerator chips / 32 TensorCore devices) using JAX/XLA forward-over-reverse automatic differentiation. We build both the RADON split-exact engine and a comprehensive peer baseline suite (AdamW, Sophia-H, AdaHessian, Distributed Shampoo).

---

## 2. Kernel Suite Architecture

### 1. RADON Split-Exact Operator (`radon/hvp.py`, `radon/split.py`)
- Forward-over-reverse JVP/VJP sweeps for exact Hessian-vector products $Hv$.
- Structural Fisher core diagonal via sampled-label gradient backpropagation (single backward, zero double-backward).
- Vectorized Sylvester-Hadamard probe generation with Latin-square tensor coloring (`radon/probes.py`).
- Trust-region coordinate-wise clipping update (`radon/optimizer.py`).

### 2. Peer Baseline Suite (`radon/baselines/`)
- `adamw.py`: First-order baseline with decoupled weight decay and cosine schedule.
- `sophia.py`: Second-order clipped stochastic curvature optimizer (Sophia-H).
- `adahessian.py`: Hutchinson-diagonal adaptive second-order optimizer.
- `shampoo.py`: Block-Kronecker preconditioned second-order optimizer.

---

## 3. TPU Acceleration Invariants
1. Purity: All computation kernels must be pure functions compatible with `@jax.jit`.
2. Hardware Topology: Support Google Cloud TPU v4-32 Pod slice (`TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"`, `TPU_HOST_BOUNDS="1,1,1"`).
3. Memory Optimization: Gradient checkpointing and fused updates to prevent TPU HBM OOM.
4. Latency Target: RADON per-step latency $\le 1.25 \times$ AdamW step time.

---

## 4. Execution & Verification Gate
```bash
python3 verify/verify_kernels.py
```
**Gate PASS Criteria:**
- All 5 optimizers initialize and execute forward, backward, and curvature step without errors.
- Forward-over-reverse HVP matches dense numerical Hessian product with relative error $< 10^{-6}$.
