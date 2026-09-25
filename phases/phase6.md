# Phase 6: Pilot Convergence & Hyperparameter Optimization

## 1. Executive Summary
Phase 6 conducts rigorous hyperparameter sweeps across all optimizers on pilot runs to ensure an unbiased, fair comparison before launching large-scale pre-training.

---

## 2. Parameter Tuning Matrix

| Parameter | Search Grid | Optimal Config |
| :--- | :--- | :--- |
| **Learning Rate $\eta$ (RADON)** | $\{1 \times 10^{-4}, 2 \times 10^{-4}, 4 \times 10^{-4}, 6 \times 10^{-4}\}$ | $4 \times 10^{-4}$ |
| **Learning Rate $\eta$ (AdamW)** | $\{3 \times 10^{-4}, 6 \times 10^{-4}, 1 \times 10^{-3}, 2 \times 10^{-3}\}$ | $1 \times 10^{-3}$ |
| **Curvature Damping $\gamma$** | $\{0.005, 0.01, 0.02, 0.05, 0.1\}$ | $0.02$ |
| **Probe Period $k$** | $\{4, 8, 16, 24, 32\}$ | $16$ |
| **Hadamard Budget $m$** | $\{4, 8, 16, 32\}$ | $16$ |
| **Trust Region Clip** | $\{0.5, 1.0, 2.0\}$ | $1.0$ |

---

## 3. Pilot Benchmark Execution
- Model: 125M Causal Transformer
- Context: 512 tokens, 4,000 steps
- Datasets: FineWeb-Edu pilot split

---

## 4. Execution & Verification Gate
```bash
python3 experiments/sweep_hparams.py
```
**Gate PASS Criteria:**
- Sweep report generated at `runs/hpo_sweep_report.json`.
- Optimal hyperparameter settings verified across all 5 optimizers.
- Confirmed RADON convergence advantage at horizon $\ge 4,000$ steps.
