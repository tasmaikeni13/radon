# Phase 2: Coded-probe statistics

For a fixed matrix \(M\), the signed probe identity is \(v_i(Mv)_i=M_{ii}+\sum_{j\ne i}M_{ij}v_iv_j\). A Hadamard cycle with one random sign vector held fixed across the cycle is unbiased for the diagonal. Its exact variance is \(\sum_{j\ne i}M_{ij}^2\bar C_{ij}^2\), where \(\bar C\) is code coherence.

Latin coloring \((a+b)\bmod m\) makes immediate row and column neighbors orthogonal, but other coordinate pairs can share codes. This is a local cancellation result, not a universal superiority claim over the same number of independent random probes. An antithetic pair \(v,-v\) returns identical diagonal products.

The next theory target is an instance-adaptive rule that selects code geometry, number of directions, and refresh timing from past observations under a cost or accuracy constraint. The existing variance identity applies to a fixed matrix and a chosen complete code cycle. A policy that selects directions or stops after seeing probes needs its own bias, variance, and stopping analysis; a cycle spread across training steps also needs a curvature-drift bound. These are open tasks, not existing Phase 2 certificates. See [theory.md](../theory.md).

Run `python3 -m verify.numerical_gate` for the 10 small fp64 checks, including exhaustive sign enumeration on a 10-coordinate example. This gate does not establish full-model variance or training performance.
