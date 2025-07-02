import torch
import torch.nn as nn
import torch.optim as optim
print()

'''1. MOMENTUM SGD'''
# momentum: 이전 업데이트의 방향&속도 고려 --> 한 방향으로 지속될때 속도 높임
# 장점: 불필요한 진동 줄임, 빠른 수렴, 지역 최소값 탈출 가능
# 단점: '모멤텀 계수' 너무 높으면 전역 최소값도 탈주할 수 있음
# 일반적인 모멘텀 계수: 0.9
model = nn.Linear(1, 1)
optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)



'''3. DROPOUT'''
# randomly drop out p*100% of input data --> prevent overfit
# scaling output: 살아남은 데이터는 1/p 배로 값 증가하여 튜런의 비활성화를 통해 손실된 "강도" 보상
dropout = nn.Dropout(p=0.5)
input_data = torch.randn(1, 6)
output = dropout(input_data)

print("input data:", input_data)
print("after applying dropout:", output)
print()