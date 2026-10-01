from torch.nn import Linear, LeakyReLU
from iterativennsimple.Sequential2D import Sequential2D, Identity
from iterativennsimple.Sequential1D import Sequential1D
import torch

from lib import get_A

get_A()
batch_size = 20
X0 = [ torch.randn(batch_size, x_size),
       torch.zeros(batch_size, y_size),
       torch.zeros(batch_size, h1_size),
       torch.zeros(batch_size, h2_size)]
Y = [ torch.zeros(batch_size, x_size),
      torch.randn(batch_size, y_size),
      torch.zeros(batch_size, h1_size),
      torch.zeros(batch_size, h2_size)]

iterations = 3
optimizer = torch.optim.Adam(A.parameters(), lr=1e-1)

for epoch in range(2000):
    X = X0
    for i in range(iterations):
        X = A.forward(X)
    loss = torch.nn.functional.mse_loss(X[1], Y[1])  # only the y slot is supervised
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:4d}  loss {loss.item():.4f}")
print(f"final      loss {loss.item():.4f}")
