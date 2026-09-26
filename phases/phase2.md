# Phase 2: Coded-probe statistics

For a fixed matrix \(M\), the signed probe identity is \(v_i(Mv)_i=M_{ii}+\sum_{j\ne i}M_{ij}v_iv_j\). A Hadamard cycle with one random sign vector held fixed across the cycle is unbiased for the diagonal. Its exact variance is \(\sum_{j\ne i}M_{ij}^2\bar C_{ij}^2\), where \(\bar C\) is code coherence.

Latin coloring \((a+b)\bmod m\) makes immediate row and column neighbors orthogonal, but other coordinate pairs can share codes. This is a local cancellation result, not a universal superiority claim over the same number of independent random probes. An antithetic pair \(v,-v\) returns identical diagonal products.

Run `python3 -m verify.numerical_gate` for the 10 small fp64 checks, including exhaustive sign enumeration on a 10-coordinate example. This gate does not establish full-model variance or training performance.
