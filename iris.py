# scp t18:/home/tiny/build/iris/outputs/2024-08-14/01-36-07/checkpoints/last.pt last_iris.pt
from tinygrad import Tensor, nn

if __name__ == "__main__":
  dat = nn.state.torch_load("last_iris.pt")
  for k,v in dat.items(): print(k, v.shape)
