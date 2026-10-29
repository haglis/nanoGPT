from sklearn.datasets import make_circles

n_samples = 1000
X, y = make_circles(n_samples,
                    noise = 0.03,
                    random_state=42)
import pandas as pd
circles = pd.DataFrame({"X1" : X[:,0], "X2" : X[:,1], "y" : y})
import matplotlib.pyplot as plt
plt.scatter(x=X[:,0], y=X[:,1])
plt.show(block=False)
import torch
X = torch.from_numpy(X).to(dtype=torch.float32)
y = torch.from_numpy(y).to(dtype=torch.long)
print(y.dtype, X.size(),len(X))

torch.manual_seed(42)
indices = torch.randperm(X.size(0))#torch.randint(0,y.size(0), (y.size(0) // 5,)).sort()
print(indices)
split = X.size(0) // 5 * 4
X_train, X_val = X[:split,:], X[split:,:]
y_train, y_val = y[:split], y[split:]
print(y_train.size(),X_train.size())
print(y_val)

import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=2)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.layer2(self.silu(self.layer1(x)))

model0 = CircleModelV0()
optimizer = torch.optim.AdamW(model0.named_parameters(), lr=0.1)

iter = 0
train_loss = 0
while iter < 5000:
    iter += 1
    model0.train()
    if iter % 100 == 0:
        model0.eval()
        with torch.inference_mode():
            logits = model0(X_val)
            loss = nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),y_val.view(y_val.size(-1)))
            print(f'train loss: {train_loss}, val loss: {loss}')
        train_loss = 0
    model0.train()
    logits = model0(X_train)
    #import pdb;pdb.set_trace()
    loss = nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),y_train.view(y_train.size(-1)))
    train_loss += loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
