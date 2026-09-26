# RADON phase protocol

The phase graph is registered in [state.json](state.json). Phases 1–5 cover proofs, statistical identities, kernels, models and data, and fp64 numerical checks. Phases 6–8 are experiment drivers; only small-model smoke tests are authorized here. Phase 9 and publication claims require measured results from the owner's heavy runs.

`GATE_PASSED` means the current verification command completed. `SMOKE_PASSED` means a small-model execution path completed. Neither status certifies training quality or TPU performance. `NOT_RUN` and `UNDER_REVIEW` identify work still pending.

```bash
python3 phases/run_phase.py --status
python3 phases/run_phase.py --phase 1
python3 phases/run_phase.py --all
```

The `--all` command runs phase gates 1–5 and smoke tests 6–8; it does not run Phase 9 or heavy training. When a phase changes, review its downstream dependencies before treating their earlier gate results as current.
