import torch
import torch.optim as optim

''' Basic Steps
1. 초기화
 - 데이터 불러오기 (x_train, y_train)
 - 가중치(weight, W, matrix), 편향(Bias, b, vector) 0으로 초기화

2. 순전파
 - 예측값 계산해서 pred 변수에 저장: pred = x_train * W + b
 - 오차 계산해서(MSE) loss 변수에 저장: loss = torch.mean((pred - y_train) ** 2)

3. 역전파
 - 각 가중치의 기울기 계산 => loss.backward() - 자동으로 해줌. 각 가중치와 편향 파라미터의 .grad에 저장
 - optimizer.step() - 계산된 기울기로 각 가중치와 편향 자동으로 업데이트 해줌.
 
4. 반복
 - 에폭(epoch) - 반복 횟수. 2, 3단계를 epoch 횟수만큼 반복.
 
5. 테스트
 - 테스트셋으로 모델 평가.
 - with torch.no_grad(): => 컨텍스트 매니저 사용해서 학습 과정에서 가중치 업데이트 방지. 기울기 계산 비활성화
 
기타
 - optimizer 설정: optimizer = optim.SGD([W, b], lr=0.01)
 - [W, b] => 어떤 파라미터들을 최적화할지 알려주는 거임.
 
'''



''' 1. 초기화 '''
# 데이터 불러오기
x_train = torch.tensor([[1], [2], [3]], dtype=torch.float)
y_train = torch.tensor([[3], [6], [9]], dtype=torch.float)

# 가중치(weight, W, matrix), 편향(Bias, b, vector) 0으로 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


''' 2. 진행시켜 '''
optimizer = optim.SGD([W, b], lr=0.01)
epochs = 1000
for epoch in range(epochs) :
    # 순전파
    pred = x_train * W + b
    loss = torch.mean((pred - y_train) ** 2)
    # 역전파
    optimizer.zero_grad() # 기울기가 누적되지 않도록! 꼭!
    loss.backward()
    optimizer.step()
    # 일정 에폭마다 학습 상태 출력해주기
    if (epoch+1) % 100 == 0 :
        print(f'Epoch [{epoch+1}/{epochs}], W: {W.item():.3f}, b: {b.item():.3f}, Loss: {loss.item():.4f}')
print()

''' 3. 모델 테스트 '''
test_x = torch.tensor([[10]], dtype=torch.float)
with torch.no_grad() :
    pred_y = test_x * W + b
    print("훈련 후 입력이 10일 때의 예측값:", pred_y.item())