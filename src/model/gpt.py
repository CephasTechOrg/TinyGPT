import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.blocks = nn.Sequential(
            *[Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)
