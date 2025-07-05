import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''1. SET OPTIMIZER AS STOCHASTIC GRADIENT DESCENT (SGD)'''
# SGD: most common/basic optimizer
model = nn.Linear(1, 1) # in_feat, out_feat
# optimizer가 optimize할 수 있게 모델의 parameter들을 넘겨줘.
optimizer = optim.SGD(params=model.parameters(), lr=0.01)



'''2. USE OPTIMIZER SGD'''
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])


print("lr = 0.01")
model_01 = nn.Linear(1, 1)
optimizer_01 = optim.SGD(params=model_01.parameters(), lr=0.01)

for epoch in range(100) : # epoch: 반복 수
    pred = model_01(x)
    loss = F.mse_loss(pred, y)
    # zero-out gradients before .backward(backpropagation)
    # b/c .backward accumulates what's done so far. that's not what we want
    optimizer_01.zero_grad()
    loss.backward() # pred.grad에 저장
    optimizer_01.step()
    if (epoch+1)%20 == 0 :
        print("Epoch", epoch+1, ", loss:", loss.item())
print()


print("lr = 0.001")
model_001 = nn.Linear(1, 1)
optimizer_001 = optim.SGD(params=model_001.parameters(), lr=0.001)

for epoch in range(100) :
    pred = model_001(x)
    loss = F.mse_loss(pred, y)
    optimizer_001.zero_grad()
    loss.backward()
    optimizer_001.step()
    if (epoch+1)%20 == 0 :
        print("Epoch", epoch, ", loss", loss.item())
print()