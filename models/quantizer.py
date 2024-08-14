from typing import List, Dict, Optional, Tuple
import math
from tinygrad import Tensor, nn
from einops import rearrange
from dataclasses import dataclass

@dataclass
class QuantizerOutput:
  q: Tensor
  tokens: Tensor
  loss: Dict[str, Tensor]
  metrics: Dict[str, float]

class Quantizer:
  def __init__(self, codebook_size: int, codebook_dim: int, input_dim: int):
    assert math.log2(codebook_size).is_integer()
    self.pre_quant_proj = nn.Linear(input_dim, codebook_dim)
    self.post_quant_proj = nn.Linear(codebook_dim, input_dim)
    self.codebook = Tensor.uniform(codebook_size, codebook_dim, low=-1.0 / codebook_size, high=1.0 / codebook_size)

  def __call__(self, z:Tensor) -> QuantizerOutput:
    z = self.pre_quant_proj(z)
    b, k = z.size(0), z.size(2)
    z = rearrange(z, 'b t k e -> (b t k) e')

    cosine_similarity = Tensor.einsum('n e, c e -> n c', z, self.codebook)
    tokens = cosine_similarity.argmax(axis=-1)  # TODO: support both axis and dim
    q = self.codebook[tokens]

    losses = {'commitment_loss': 0.02 * (z - q.detach()).pow(2).mean()}
    metrics = {}

    q = z + (q - z).detach()
    q = self.post_quant_proj(q)

    q = rearrange(q, '(b t k) e -> b t k e', b=b, k=k)
    tokens = rearrange(tokens, '(b t k) -> b t k', b=b, k=k)

    return QuantizerOutput(q, tokens, losses, metrics)

  def embed_tokens(self, tokens: Tensor) -> Tensor:
    return self.post_quant_proj(self.codebook[tokens])
