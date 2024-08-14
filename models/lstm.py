from tinygrad.tensor import Tensor

# we don't have this in our nn library. TODO: add it?
class LSTMCell:
  def __init__(self):
    self.weight_ih = Tensor.zeros(2048, 1024)
    self.weight_hh = Tensor.zeros(2048, 512)
    self.bias_ih = Tensor.zeros(2048)
    self.bias_hh = Tensor.zeros(2048)
  def __call__(self, x:Tensor, h:Tensor, c:Tensor) -> Tensor:
    gates = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
    i, f, g, o = gates.split([512, 512, 512, 512], dim=1)  # TODO: axis
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    new_c = f * c + i * g
    new_h = o * new_c.tanh()
    # contiguous prevents buildup
    return (new_h.contiguous(), new_c.contiguous())
