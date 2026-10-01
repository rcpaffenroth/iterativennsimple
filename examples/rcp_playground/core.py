from torch.nn import Linear, LeakyReLU
from iterativennsimple.Sequential2D import Sequential2D, Identity
from iterativennsimple.Sequential1D import Sequential1D
import torch

torch.manual_seed(0)

x_size = 10
y_size = 8 
h1_size = 5
h2_size = 7

# blocks[i][j] maps input slot i to output slot j, i.e. it is the TRANSPOSE
# of the block matrix acting on a column vector [x; y; h1; h2].
# Slots: 0 = x, 1 = y, 2 = h1, 3 = h2.
blocks = [[None] * 4 for _ in range(4)]
blocks[0][0] = Identity(in_features=x_size, out_features=x_size)  # x  -> x  (keep the input clamped)
# x -> y
blocks[0][1] = Linear(x_size, y_size, bias=False) 
# x -> h1
blocks[0][2] = Linear(x_size, h1_size, bias=False) 
# h1 -> h1
blocks[2][2] = Sequential1D(Linear(h1_size, h1_size, bias=False), 
                            LeakyReLU(),
                            in_features=h1_size, out_features=h1_size)
# h1 -> y
blocks[2][1] = Linear(h1_size, y_size, bias=False) 


A = Sequential2D([x_size, y_size, h1_size, h2_size],
                 [x_size, y_size, h1_size, h2_size],
                 blocks)

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
