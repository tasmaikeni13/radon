# Phase 3: Curvature and optimizer kernels

`radon/hvp.py` computes exact PyTorch Hessian-vector products by reverse-over-reverse autodiff and exposes a JAX forward-over-reverse helper when JAX is installed. `radon/split.py` computes sampled structural diagonals and residual products with PyTorch autograd. `radon/probes.py` creates Hadamard-coded sign directions; `radon/optimizer.py` applies clipped diagonal preconditioning. CPU tests exercise these paths.

`radon/tpu.py` contains optional torch_xla device and collective wrappers. No physical TPU is attached in the local verification environment, so multi-host execution, throughput, and communication correctness remain unverified. The v4-32 target is 16 chips and 32 TensorCores across 4 hosts, according to Google Cloud's v4 topology table.

Sophia-H and AdaHessian now use autodiff Hutchinson samples. Their curvature samples are averaged across workers before the moving-average update, matching the shared gradient synchronization. Shampoo uses blocked matrix preconditioners for large matrices and a diagonal fallback only for vectors. These CPU checks do not establish method-level parity or performance on TPU hardware.

Run `python3 -m verify.verify_kernels` and `pytest tests/` for the current CPU checks. These are software smoke checks, not hardware certification.
