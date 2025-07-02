import torch
import torch.nn as nn
import torch.nn.functional as F

'''1. GRADIENT OF MSE'''
actual_value = torch.tensor([2.0])
prediction1 = torch.tensor([3.0], requires_grad=True)
prediction2 = torch.tensor([10.0], requires_grad=True)

mse_loss1 = F.mse_loss(prediction1, actual_value)
mse_loss2 = F.mse_loss(prediction2, actual_value)

mse_loss1.backward() # prediction1.grad에 저장
mse_loss2.backward() # prediciton2.grad에 저장

pred_1_gradient = prediction1.grad.item()
pred_2_gradient = prediction2.grad.item()

print("첫 번째 예측값의 MSE 손실:", mse_loss1)
print("두 번째 예측값의 MSE 손실:", mse_loss2)
print("첫 번째 예측값의 MSE 손실 함수 미분값:", pred_1_gradient)
print("두 번째 예측값의 MSE 손실 함수 미분값:", pred_2_gradient)
print()
print()



'''2. GRADIENT OF BCE'''
actual_label = torch.tensor([0.0])
prediction1 = torch.tensor([1.0], requires_grad=True)
prediction2 = torch.tensor([3.0], requires_grad=True)

prob1 = torch.sigmoid(prediction1)
prob2 = torch.sigmoid(prediction2)

bce_loss = nn.BCELoss()
bce_loss1 = bce_loss(prob1, actual_label)
bce_loss2 = bce_loss(prob2, actual_label)

bce_loss1.backward() # prediction1.grad에 저장
bce_loss2.backward() # prediction2.grad에 저장

pred1_gradient = prediction1.grad.item()
pred2_gradient = prediction2.grad.item()

print("첫 번째 예측값의 BCE 손실:", bce_loss1)
print("두 번째 예측값의 BCE 손실:", bce_loss2)
print("첫 번째 예측값의 BCE 손실 함수 미분값:", pred1_gradient)
print("두 번째 예측값의 BCE 손실 함수 미분값:", pred2_gradient)
print()
print()



'''3. GRADIENT CCE'''
target = torch.tensor([0])
logits = torch.tensor([0.1, 0.2, 0.7], requires_grad=True)

cross_entropy_loss = nn.CrossEntropyLoss()

loss = cross_entropy_loss(logits.unsqueeze(0), target) # 배치 차원을 추가
loss.backward() # logits.grad에 저장

logits_gradient = logits.grad
print("로짓 세트의 CCE 손실 함수 미분값:", logits_gradient)
print()