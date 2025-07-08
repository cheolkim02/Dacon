import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)

x_train = torch.tensor([[10, 1], # 특성: 방의 크기 (size), 방의 개수 (rooms)
                        [15, 2]], dtype=torch.float)
y_train = torch.tensor([[40], # 타겟: 주택 가격 (price)
                        [70]], dtype=torch.float)

''' 1. 순정 '''
# 2. 가중치, 편향 초기화 => y=x1w1 + x2w2 + b
W = torch.zeros(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.001)

epochs = 10000
for epoch in range(epochs) :
    pred = x_train.mm(W.unsqueeze(1)) + b
    loss = torch.mean((pred - y_train) ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 3. test
test_x = torch.tensor([[20, 3]], dtype=torch.float)  # 예: 20제곱미터 크기에 방 3개인 집
with torch.no_grad():  
    pred_y = test_x.mm(W.unsqueeze(1)) + b
    print(f"새로운 데이터에 대한 예측 가격: {pred_y.item()}")
    
    
    
''' 2. nn.Linear 사용 '''
# 입력 및 출력 특성의 크기 계산
in_features = x_train.shape[1]
out_features = y_train.shape[1]

# 선형 회귀 모델 초기화
model = nn.Linear(in_features=2, out_features=1)
for param in model.parameters():
    print(param.data)

import torch.optim as optim
import torch.nn.functional as F

optimizer = optim.SGD(model.parameters(), lr=0.001)
epochs = 10000
for epoch in range(epochs):
    pred = model(x_train)
    loss = F.mse_loss(pred, y_train)

    optimizer.zero_grad()  
    loss.backward()        
    optimizer.step()       

    if (epoch+1) % 100 == 0:
         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

test_x = torch.tensor([[20, 3]], dtype=torch.float)  # 예: 120제곱미터 크기에 방 3개인 집
with torch.no_grad():  
    pred_y = model(test_x)
    print(f"새로운 데이터에 대한 예측 가격: {pred_y.item()}")



''' 3. 객체로 만들기 '''
class LinaerRegressionModel(nn.Module) :
    def __init__(self):
        super(LinaerRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=1)
    def forward(self, x) :
        return self.linear(x)
model = LinaerRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=0.001)
nb_epochs = 10000
for epoch in range(nb_epochs):
    pred = model(x_train)
    loss = F.mse_loss(pred, y_train)

    optimizer.zero_grad()  
    loss.backward()        
    optimizer.step()       

    if (epoch+1) % 100 == 0:
         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

new_x = torch.tensor([[12, 3]], dtype=torch.float)  
with torch.no_grad(): 
    predicted_price = model(new_x)
    print(f"새로운 데이터에 대한 예측 가격: {predicted_price.item()} 만 달러")