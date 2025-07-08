import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
torch.manual_seed(0)

# 1. set data
input = torch.tensor([[3, 0.31, 22.6, 11.7],
                             [1, 2.48, 13.5, 7.52],
                             [3, 1.52, 18.9, 17.1]], dtype=torch.float32)
target = torch.tensor([[307], [110], [369]], dtype=torch.float32)

# 2. make dataset and data loader
# TensorDataset. 텐서를 감싸는 Dataset. 입력&타겟 텐서 결합.
dataset = TensorDataset(input, target)
# DataLoader. dataset의 data를 iterable하게 생성
# batch_size=1. each mini-batch has one data point. 한 반복마다 하나의 데이터 포인트로만 학습함.
# shuffle=True. 에폭마다 데이터셋 섞어서 데이터 순서 랜덤하게 바꿈. 과적합 방지.
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 3. make class
class SimpleMLP(nn.Module) :
    # instance가 생성될 때 한 번 호출. 인풋 수 받음.
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        # 은닉층 정의. 선형으로 조합. (입력층의 크기(넓이), 은닉층의 뉴런 수)
        self.hidden = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 1) # 출력층 정의. (은닉층 뉴런 수, 출력층 뉴런 수)
    # 순전파 수행될 때 호출. x는 모델의 입력 데이터.
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

learning_rate = 0.001
epochs = 1000
model = SimpleMLP(input_size=input.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_history = []

for epoch in range(epochs):
    for inputs, targets in data_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

plt.plot(loss_history)
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()