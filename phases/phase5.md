# Phase 5: Numerical exactness checks

`python3 -m verify.numerical_gate` runs ten fp64 checks on a small neural network. It compares the dense Hessian with independently constructed structural and residual terms; checks the residual HVP implementation against dense curvature; checks structural positive semidefiniteness; and checks coded recovery, coded variance, sampled Fisher unbiasedness, and ideal diagonal Newton algebra.

The tolerances are recorded in `verify/numerical_gate.py`. A pass covers these examples and identities. It does not establish full-model training performance, TPU correctness, or the variance of the implemented sampled structural channel.
