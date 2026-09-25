"""Vision Transformer (ViT-Small/16) Architecture.

Pre-norm, GELU activations, scaled dot-product attention.
GELU ensures C² smoothness, required for exact second derivatives.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    n_classes: int = 1000
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class Attention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig | None = None):
        super().__init__()
        self.cfg = cfg or ViTConfig()
        n_patches = (self.cfg.img_size // self.cfg.patch_size) ** 2
        patch_dim = self.cfg.in_channels * (self.cfg.patch_size**2)

        self.patch_embed = nn.Linear(patch_dim, self.cfg.n_embd, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, self.cfg.n_embd))
        self.drop = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layer)])
        self.ln_f = LayerNorm(self.cfg.n_embd)
        self.head = nn.Linear(self.cfg.n_embd, self.cfg.n_classes, bias=False)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.cfg.patch_size
        patches = x.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * P * P)

        x_emb = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)
        x_seq = self.drop(x_seq + self.pos_embed)

        for block in self.blocks:
            x_seq = block(x_seq)

        x_cls = self.ln_f(x_seq[:, 0])
        return self.head(x_cls)
