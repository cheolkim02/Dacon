import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
nn.Linear 모듈로 가중치와 편향을 자동으로 관리하는 모델 구현.
nn.Module 클래스로 사용자 정의 모델을 만드는 두 가지 방식 탐구.
->모델의 구조화 및 모듈화 방법을 학습하고, PyTorch의 다양한 최적화 기법을 적용하는 방법을 실습
'''

''' 1. BUILD LIN REG MODEL USING nn.Linear'''
# 1. prepare data
# returns torch.Generator object. used for random samples.
# PyTorch에서 난수 생성의 재현성을 보장하기 위해 사용되는 함수
torch.manual_seed(42)


x_train = torch.tensor([[1], [2], [3]], dtype=torch.float)
y_train = torch.tensor([[3], [6], [9]], dtype=torch.float)

# 2. initialize model and set params
in_features = x_train.shape[1]
out_features = y_train.shape[1]
print("in features:", in_features)
print("out features:", out_features) # 이 프린트를 해서 밑에 두 개가 1임을 확인한거임.
model = nn.Linear(in_features=1, out_features=1) # 모델 초기화

# 3. initialize parameters - 파라이머 초기화
# nn.Linear로 모델 만들 때 이미 된거임. 무작위로.
for param in model.parameters() :
    print(param.data)
    
# 4. 선형 회귀 모델 학습
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
    # 일정 에폭마다 학습 상태 출력해주지
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1:4d}/{nb_epochs}] W: {model.weight.item():.3f}, b: {model.bias.item():.3f} loss: {loss.item():.3f}')
        
# 5. 모델 테스트
test_x = torch.tensor([[10]], dtype=torch.float)
with torch.no_grad():
    pred_y = model(test_x)
    print("훈련 후 입력이 10일 때의 예측값 :", pred_y.item())
    


''' 2. nn.Module을 상속으로 구현한 사용자 정의 선형 회귀 클래스 '''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. 데이터 준비, 모델 초기화

x_train = torch.tensor([[1], [2], [3]], dtype=torch.float)
y_train = torch.tensor([[3], [6], [9]], dtype=torch.float)

class MyLinearModel(nn.Module) :
    def __init__(self):
        super(MyLinearModel, self).__init__()
        # 선형 레이어 정의, in/out features 1로 고정
        self.linear = nn.Linear(in_features=1, out_features=1)
    def forward(self, x) :
        # 입력 x에 대해 선형 변환 수행
        return self.linear(x)

model = MyLinearModel()

#2. 선형 회귀 모델 학습
optimizer = optim.SGD(model.parameters(), lr=0.01)
nb_opochs = 1000
for epoch in range(nb_epochs) :
    # forward
    pred = model(x_train)
    lost = F.mse_loss(pred, y_train)
    # backward
    optimizer.zero_grad()
    lost.backward()
    optimizer.step()
    # check
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} W: {model.linear.weight.item():.3f}, b: {model.linear.bias.item():.3f} Loss: {loss.item():.6f}')

# 3. 테스트
test_x = torch.tensor([[10]], dtype=torch.float)
with torch.no_grad():
    pred_y = model(test_x)
    print("훈련 후 입력이 10일 때의 예측값 :", pred_y.item())