# implementation of GPT-2 124M model in pytorch inspired by Karpathy

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


# model config dataclass
@dataclass
class GPTConfig:
    block_size: int = 100
    vocab_size: int = 50257
    n_embed: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.0


# Transformer block
class Block(nn.Module):
    """A single transformer block"""
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttetion(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        # x shape is (B, T, C)
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))
