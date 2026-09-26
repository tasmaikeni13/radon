# Phase 4: Models and data

The default causal transformer in `models/transformer.py` has 12 decoder blocks, width 768, 12 attention heads, context length 2048, tied token and output weights, RMSNorm, learned absolute position embeddings, and 125,160,192 parameters. `models/vit.py` provides a 12-layer, width-384 ViT-Small/16.

`data/fineweb.py` loads memory-mapped NumPy token arrays. The offline synthetic fallback is for tests only. A full-run driver must require real data and report the actual file, token count, and validation split.

Run `python3 -m verify.verify_models` on CPU to check both models' forward and backward passes and a data batch. This does not test TPU throughput or distributed loading.
