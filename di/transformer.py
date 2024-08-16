from typing import Union
from tinygrad import Tensor, nn, Variable
from einops import rearrange

MAX_CONTEXT = 156
EMBED_DIM = 256

class MLPLayer:
  def __init__(self):
    self.ln = nn.LayerNorm(EMBED_DIM)
    self.mlp = [nn.Linear(EMBED_DIM, 4*EMBED_DIM), Tensor.gelu, nn.Linear(4*EMBED_DIM, EMBED_DIM)]
  def __call__(self, x:Tensor): return x + self.ln(x).sequential(self.mlp)

class Attention:
  def __init__(self):
    self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
    self.num_heads = 4
    self.start_pos = 0

  def __call__(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
    # update the cache
    """
    #if not hasattr(self, 'cache_kv'): self.cache_kv = Tensor.zeros(2, k.size(0), MAX_CONTEXT, EMBED_DIM, dtype=k.dtype).contiguous().realize()
    seqlen = q.size(1)
    assert k.dtype == v.dtype == self.cache_kv.dtype, f"{k.dtype=}, {v.dtype=}, {self.cache_kv.dtype=}"
    self.cache_kv.shrink((None, None, (self.start_pos, self.start_pos+seqlen), None)).assign(Tensor.stack(k, v)).realize()
    k = self.cache_kv[0].shrink((None, (0, self.start_pos+seqlen), None))
    v = self.cache_kv[1].shrink((None, (0, self.start_pos+seqlen), None))
    self.start_pos += seqlen
    """

    q: Tensor = rearrange(q, 'b q (h e) -> b h q e', h=self.num_heads)
    k = rearrange(k, 'b k (h e) -> b h k e', h=self.num_heads)
    v = rearrange(v, 'b k (h d) -> b h k d', h=self.num_heads)
    y = q.scaled_dot_product_attention(k, v, is_causal=True)
    y = rearrange(y, 'b h q d -> b q (h d)')
    return self.proj(y)

class SelfAttentionLayer:
  def __init__(self):
    self.ln = nn.LayerNorm(EMBED_DIM)
    self.query = nn.Linear(EMBED_DIM, EMBED_DIM)
    self.key = nn.Linear(EMBED_DIM, EMBED_DIM)
    self.value = nn.Linear(EMBED_DIM, EMBED_DIM)
    self.attention = Attention()
  def __call__(self, inputs:Tensor) -> Tensor:
    x = self.ln(inputs)
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    return inputs + self.attention(q, k, v)

class EncoderLayer:
  def __init__(self):
    self.sa = SelfAttentionLayer()
    self.mlp = MLPLayer()
  def __call__(self, x:Tensor) -> Tensor: return self.mlp(self.sa(x))

class TransformerEncoder:
  def __init__(self):
    self.pos_emb = nn.Embedding(MAX_CONTEXT, EMBED_DIM)
    self.ln = nn.LayerNorm(EMBED_DIM)
    self.blocks = [EncoderLayer() for _ in range(3)]
  def __call__(self, x:Tensor) -> Tensor:
    assert x.ndim == 3 and x.size(2) == EMBED_DIM # (B, TK, E)
    y = x + self.pos_emb(Tensor.arange(x.size(1)))
    return self.ln(y.sequential(self.blocks))

class Head:
  def __init__(self, output_dim):
    self.head_module = [
      nn.Linear(EMBED_DIM, EMBED_DIM), Tensor.relu,
      nn.Linear(EMBED_DIM, output_dim)]
  def __call__(self, outputs:Tensor):
    return outputs.sequential(self.head_module)
