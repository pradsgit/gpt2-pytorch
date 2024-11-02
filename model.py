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

# causal masked self-attention layer
class CausalSelfAttention(nn.Module):
    """A multi-head masked self-attention layer"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0, "n_embed must be divisible by n_head"
        self.qkv = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        # causal mask to ensure that the attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # shape of x is (B, T, C)
        B, T, C = x.size()
        assert C == self.n_embed, f"hidden dim of input(C) must be equal to n_embed, but got {C} and {self.n_embed}"

        qkv = (
            self.qkv(x) # (B, T, C * 3)
            .reshape(B, T, 3, self.n_head, C // self.n_head) # (B, T, 3, nh, hs)
            .permute(2, 0, 3, 1, 4) # (3, B, nh, T, hs)
        )
        q, k, v = qkv.unbind(0) # q, k, v shape is (B, nh, T, hs)

        # # project input to qkv matrix
        # qkv = self.c_attn(x) # qkv shape is (B, T, C * 3)
        # # split qkv into q, k, v
        # q, k, v = qkv.split(self.n_embed, dim=-1) # each of q, k, v shape is (B, T, C)
        # # reshape q, k, v with shape (B, T, C) to (B, nh, T, hs) multi-head
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # q shape is (B, nh, T, hs)
        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # k shape is (B, nh, T, hs)
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # v shape is (B, nh, T, hs)

        # calculate scaled alignment scores
        align_scores = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1))) # align_scores shape is (B, nh, T, T)
        # apply mask
        align_scores = align_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # apply mask
        # softmax
        align_scores = F.softmax(align_scores, dim=-1) # align_scores shape is (B, nh, T, T)
        # apply attention
        x = align_scores @ v # x shape is (B, nh, T, hs)
        # reshape x with shape (B, nh, T, hs) to (B, T, C)
        x = x.transpose(1, 2).contiguous().view(B, T, C) # x shape is (B, T, C)
        # project x to output
        x = self.c_proj(x) # x shape is (B, T, C)
        return x

# mlp layer
class MLP(nn.Module):

    """A feed-forward neural network. these are very good pattern learners"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
    
    def forward(self, x):
        # shape of x is (B, T, C)
        return self.fc2(self.gelu(self.fc1(x)))


# Transformer block
class Block(nn.Module):
    """A single transformer block"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        # x shape is (B, T, C)
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


# model
class GPT2(nn.Module):
    """A 124M parameter GPT2 model"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embed) # token embedding lookup table
        self.wpe = nn.Embedding(config.block_size, config.n_embed) # position embedding lookup table
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

    def forward(self, x):
        # x shape is (B, T)
        B, T = x.shape
        # token embedding
        tok_emb = self.wte(x) # tok_emb shape is (B, T, C)
        # absolute position embeddings
        pos_emb = self.wpe(torch.arange(T, device=x.device)) # pos_emb shape is (T, C)
        # add token and position embeddings 
        x = tok_emb + pos_emb # x shape is (B, T, C)
        # apply transformer blocks
        logits = self.blocks(x) # x shape is (B, T, C)
        return logits