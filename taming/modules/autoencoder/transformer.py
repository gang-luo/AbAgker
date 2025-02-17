import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 1, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        self.feed_forward = FeedForward(dim, ff_dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, memory, tgt_mask, memory_mask):
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2, tgt_mask)
        x2 = self.norm2(x)
        x = x + self.cross_attn(x2, memory, memory, memory_mask)
        x2 = self.norm3(x)
        x = x + self.feed_forward(x2)
        return x

