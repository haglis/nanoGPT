import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
        self.loss = nn.L1Loss()

    def forward(self, x: torch.tensor, y=None) -> torch.tensor:
        y_pred = self.weights * x + self.bias
        if y==None:
            return y_pred
        else:
            return y_pred, self.loss(y_pred, y)

weights = 0.7
bias = 0.3

data = torch.arange(start=0, end=100, step=0.2)
labels = weights*data + bias

train_end = int(0.8*len(data))
x_train, y_train = data[:train_end], labels[:train_end]
x_test, y_test = data[train_end:], labels[train_end:]


def print_train_val_predit(x_train, y_train, x_test, y_test, y_pred=None):
    #fig, ax = plt.subplots()
    plt.figure(figsize=(8,3))
    plt.scatter(x_train, y_train,c='b')

    plt.scatter(x_test, y_test, c='r')

    if y_pred is not None:
        plt.scatter(x_test, y_pred, c='y')

    plt.show(block=False)

model = LinearRegressionModel()
print(list(model.named_parameters()))
optimizer = torch.optim.AdamW(model.named_parameters(),lr=0.01)
iter = 0
while iter < 5000:
    iter += 1
    if iter % 100 == 0:
        model.eval()
        with torch.inference_mode():
            model.eval()
            y_pred = model(x_test)
            print_train_val_predit(x_train, y_train, x_test, y_test, y_pred.detach())
        model.train()
        import pdb;pdb.set_trace()
    optimizer.zero_grad()
    y_pred, loss = model(x_train, y_train)
    loss.backward()
    optimizer.step()
    y_pred = model(x_test)
    
    


y_pred = model(x_train)

