# Phase 4: Frontier Architectures & Data Ingestion Pipeline

## 1. Executive Summary
Phase 4 implements the target model architectures and streaming dataset pipelines. Specifically, it establishes a 124.5M Causal Transformer matching GPT-2 Small standards and a Vision Transformer (ViT-Small/16), together with a high-throughput streaming data pipeline for FineWeb-Edu.

---

## 2. Architecture Specifications

### 124.5M Causal Transformer (`models/transformer.py`)
- Parameters: 124,534,272
- Layers: 12 decoder blocks
- Hidden Dimension: $d = 768$
- Attention Heads: 12 ($d_k = 64$)
- Feedforward Dimension: $4 \times d = 3072$
- Context Length: 2048 tokens
- Positional Encoding: Rotary Position Embeddings (RoPE)
- Activation: GELU (smooth $C^2$ everywhere)
- Vocabulary Size: 50,257 (GPT-2 BPE)
- Tied Weights: Input and output embedding matrices tied

### Vision Transformer ViT-Small/16 (`models/vit.py`)
- Patch size: $16 \times 16$
- Hidden dimension: 384, 12 layers, 6 heads
- Classification head: 100/1000 classes

---

## 3. Data Pipeline: FineWeb-Edu (`data/fineweb.py`)
- Dataset: FineWeb-Edu (high-educational-quality web crawl)
- Sharding & Streaming: Deterministic pseudo-random chunk streaming with high-speed memory caching
- Hermetic Synthetic Fallback: Automated generation of synthetic text batches for offline verification and regression testing without network dependencies.

---

## 4. Execution & Verification Gate
```bash
python3 verify/verify_models.py
```
**Gate PASS Criteria:**
- 125M Transformer forward and backward passes execute cleanly.
- Parameter count verified: $124.5\mathrm{M} \pm 0.5\mathrm{M}$.
- FineWeb-Edu pipeline yields valid $(B, T)$ token batches with zero NaNs.
