"""Verification suite for model architectures (125M Causal Transformer, ViT) and dataset streaming."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.fineweb import FineWebDataset
from models.transformer import CausalTransformer, TransformerConfig
from models.vit import VisionTransformer, ViTConfig


def test_causal_transformer():
    print("Testing 124.5M Causal Transformer...")
    cfg = TransformerConfig(
        vocab_size=50304,
        block_size=2048,
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    model = CausalTransformer(cfg)
    n_params = model.get_num_params()
    print(f"  Total parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    assert 120_000_000 <= n_params <= 130_000_000, f"Expected ~125M params, got {n_params}"

    # Forward pass smoke test
    x = torch.randint(0, 50257, (2, 64))
    y = torch.randint(0, 50257, (2, 64))
    logits, loss = model(x, y)
    assert logits.shape == (2, 64, 50304)
    assert loss is not None and not torch.isnan(loss)
    print(f"  Forward pass successful: logits shape={logits.shape}, loss={loss.item():.4f}")
    return True


def test_vision_transformer():
    print("Testing Vision Transformer (ViT-Small/16)...")
    cfg = ViTConfig(
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_classes=1000,
        n_layer=12,
        n_head=6,
        n_embd=384,
    )
    model = VisionTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Forward pass smoke test
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1000)
    assert not torch.isnan(out).any()
    print(f"  Forward pass successful: out shape={out.shape}")
    return True


def test_fineweb_pipeline():
    print("Testing FineWeb-Edu pipeline...")
    dataset = FineWebDataset(is_train=True)
    device = torch.device("cpu")
    x, y = dataset.get_batch(batch_size=4, block_size=128, device=device, seed=42)
    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    print(f"  FineWeb batch fetched: shape={x.shape}, synthetic={dataset.synthetic}, tokens={dataset.num_tokens:,}")
    return True


def main():
    print("=" * 80)
    print("  RADON Model Architectures and Data Pipeline Verification")
    print("=" * 80)

    test_causal_transformer()
    test_vision_transformer()
    test_fineweb_pipeline()

    print("=" * 80)
    print("[SUCCESS] All model architectures and data pipelines verified!")


if __name__ == "__main__":
    main()
