
'''1. APPLY SIGMOID FUNCTION'''
# sigmoid: 이진 분류. 각 값을 0과 1 사이의 확률 값으로 변환
import torch
# 데이터에 sigmoid 함수 적용시키기
torch_data = torch.tensor([-1.0, 0.0, 1.0])
sigmoid_torch_result = torch.sigmoid(torch_data)
print("sigmoid applied tensor:", sigmoid_torch_result)
print()


'''1-2. MAKE LINEAR LAYER AND APPLY SIGMOID ACTIVATION FUNCTION'''
import torch
import torch.nn as nn
# 입력값 2개, 출력값 1개인 linear layer 만들기
linear_layer = nn.Linear(in_features=2, out_features=1)

# create random tensor with size (1, 2)
# size: batch size = 1, input features = 2
input_tensor = torch.randn(1, 2)
print("input tensor:", input_tensor)

# make it go through lineary layer
linear_output = linear_layer(input_tensor)
print("linear output:", linear_output)

# apply sigmoid activation function
sigmoid_output = torch.sigmoid(linear_output)
print("sigmoid output:", sigmoid_output)
print()
print()



'''2. APPLY RELU FUNCTION'''
import torch
x = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0])
relu_x = torch.relu(x)
print("relu x:", relu_x)
print()


'''2-1. MAKE LINEAR LAYER AND APPLY RELU ACTIVATION FUNCTION'''
import torch
import torch.nn as nn
linear_layer = nn.Linear(in_features=5, out_features=1) # make linear layer
input_tensor = torch.randn(1, 5) # make random tensor of size(1, 5)
linear_output = linear_layer(input_tensor)
relu_output = torch.relu(linear_output)
print("input tensor:", input_tensor)
print("linear output:", linear_output)
print("relu output:", relu_output)
print()
print()


'''3. APPLY SOFTMAX FUNCTION'''
# softmax: 다중 분류. 각 클래스에 대한 확률 제시.
import torch
logits = torch.tensor([2.0, 1.0, 0.1]) # logit: raw values b4 anything
softmax_result = torch.softmax(logits, dim=0) # dim: softmax가 적용될 차원.
print("softmax result:", softmax_result)
print()


'''3-1 MAKE LINEARY LEAYER AND APPLY SOFTMAX ACTIVATION FUNCTION'''
import torch
import torch.nn as nn
linear_layer = nn.Linear(in_features=3, out_features=1)
input_tensor = torch.randn(1, 3)
linear_output = linear_layer(input_tensor)
softmax_output = torch.softmax(linear_output, dim=1)
print("input tensor:", input_tensor)
print("linear output:", linear_output)
print("softmax output", softmax_output)
print()
print()