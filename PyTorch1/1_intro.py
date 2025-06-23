import torch

tensor = torch.tensor([1, 2, 3])
print(tensor)

# 이렇게 직관적이고 쉽다~
model = torch.nn.Sequential (
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2)
)

# cuda GPU 사용 가능하다~
if torch.cuda.is_available() :
    tensor = tensor.to('cuda')

# 자동 미분 가능하다~
x = torch.tensor(1., requires_grad=True)
y = x*2
y.backward() # 미분 계산
print(x.grad) # x에 대한 y의 미분값 출력 (dy/dx)

# 모듈화 및 재사용 가능성
class CustomLayer(torch.nn.Module) :
    # 레이어 초기화
    def __init__(self) :
        super(CustomLayer, self).__init__()
    # 데이터 x에 대한 연산
    def forward(self, x) :
        return x
# 모델에 사용자 정의 레이어 적용
model.add_module('custom_layer', CustomLayer())

# pytorch 예시 코드
a = torch.tensor([2., 3.])
b = torch.tensor([6., 4.])
c = torch.add(a, b)
print(c)

# pytorch 핵심 모듈
'''
torch
- import torch
torch.nn
- import torch.nn as nn
torch.utils.data
- from torch.utils.data import data
torch.optim
- import torch.optim as optim
torch.autograd
- import torch.autograd as autograd
torch.cuda
- import torch.cuda as cuda
'''