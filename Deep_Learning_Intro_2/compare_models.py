import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

x_train = torch.tensor([[1], [2], [3]], dtype=torch.float)
y_train = torch.tensor([[3], [6], [9]], dtype=torch.float)

''' 1. no nn '''
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)
epochs = 1000
for epoch in range(epochs) :
    # 순전파
    pred = x_train * W + b
    loss = torch.mean((pred - y_train) ** 2)
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

''' 2. use nn.Linear '''
in_features = x_train.shape[1]
out_features = y_train.shape[1]
print("in features:", in_features)
print("out features:", out_features)
model = nn.Linear(in_features=1, out_features=1)
    
optimizer = optim.SGD(model.parameters(), lr=0.01)
nb_epochs = 1000
for epoch in range(nb_epochs) :
    # 순전파
    pred = model(x_train)
    loss = F.mse_loss(pred, y_train)
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

''' 3. make class '''
class MyLinearModel(nn.Module) :
    def __init__(self):
        super(MyLinearModel, self).__init__()
        # 선형 레이어 정의, in/out features 1로 고정
        self.linear = nn.Linear(in_features=1, out_features=1)
    def forward(self, x) :
        # 입력 x에 대해 선형 변환 수행
        return self.linear(x)
model = MyLinearModel()

optimizer = optim.SGD(model.parameters(), lr=0.01)
nb_opochs = 1000
for epoch in range(nb_epochs) :
    # 순전파
    pred = model(x_train)
    cost = F.mse_loss(pred, y_train)
    # 역전파
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
