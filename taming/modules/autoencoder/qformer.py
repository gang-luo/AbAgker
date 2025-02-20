from torch import nn
from einops import rearrange
from .transformer import MultiHeadAttention
from .transformer import FeedForward


class QLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        self.feed_forward = FeedForward(dim, ff_dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, memory):
        x2 = self.norm1(x)
        x = x + self.cross_attn(x2, memory, memory, None)
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
        return x
    
class Q_former(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, num_layers: int,nums_query: int,m_len: int):
        super().__init__()
        self.layers = nn.ModuleList([QLayer(d_model, num_heads, ff_dim) for _ in range(num_layers)])
        self.masked_pre_fc = nn.Linear(nums_query, m_len)  # 63 -> 128
        self.gelu = nn.GELU() 
    
    def forward(self, x, memory):
        ### x为DTA embeding; memory为protein related info for Mols generation
        for layer in self.layers:
            x = layer(x, memory)
        
        x = rearrange(x , 'b l i -> b i l').contiguous()
        x = self.gelu(self.masked_pre_fc(x))
        x = rearrange( x , 'b i l -> b l i').contiguous()

        return x
