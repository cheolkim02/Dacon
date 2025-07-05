
'''1. 가중치 조정을 통한 신경망의 예측 오차 최소화'''
print()
input_data = [30, 2] # 입력 데이터
target_actual = 3000  # 실제 목표값
weights_0 = {'node_0': [100, 10]} # 첫 번째 가중치 설정

# 첫 번째 가중치를 사용한 예측값
output_0 = input_data[0] * weights_0['node_0'][0] + input_data[1] * weights_0['node_0'][1]
error_0 = output_0 - target_actual # 오차

# 가중치 조정 및 두 번째 예측
weights_1 = {'node_0': [90, 150]} # 두 번째 가중치 설정 (가중치 조정)
output_1 = input_data[0] * weights_1['node_0'][0] + input_data[1] * weights_1['node_0'][1]
error_1 = output_1 - target_actual

# print("첫 번째 예측 오차:", error_0)
# print("조정된 가중치로의 예측 오차:", error_1)
# print()
# print()



'''2. ReLU를 활용한 가중치 업데이트'''
def relu_derivative(output) :
    return 1 if output>0 else 0

predicted_value = 15
target_value = 18
error = predicted_value - target_value

error_gradient = 2*error

hidden_node_output = 152 # 그렇다 칩시다

# 가중치에 대한 손실 함수의 기울기
hidden_node_gradient = relu_derivative(hidden_node_output)
weight_gradient = hidden_node_output * error_gradient * hidden_node_gradient

# 다음 가중치 계산하기
learning_rate = 0.01  
next_weight = 0.1 - learning_rate * weight_gradient

# print("조정된 가중치 값:", next_weight)
# print()
# print()



'''3. 역전파로 집 가격 예측 모델 구현''' '''요기가 약간 진짜임! 오차역전파 구현'''
def relu(x):
    return max(0, x)

def relu_derivative(output):
    return 1 if output > 0 else 0

input_data = [30.0, 2.0]
weights = {
    'hidden1_node0': [50.0, 10.0],
    'hidden1_node1': [-1.0, 0.0],
    'hidden2_node0': [0.1, 10.0],
    'hidden2_node1': [-0.1, 0.0],
    'output': [0.1, 25.0]
}

### 순전파 ###
# First hidden layer
hidden1_node0_output = input_data[0] * weights['hidden1_node0'][0] + input_data[1] * weights['hidden1_node0'][1]
hidden1_node0_activated = relu(hidden1_node0_output)
hidden1_node1_output = input_data[0] * weights['hidden1_node1'][0] + input_data[1] * weights['hidden1_node1'][1]
hidden1_node1_activated = relu(hidden1_node1_output)

# Second hidden layer
hidden2_node0_output = hidden1_node0_activated * weights['hidden2_node0'][0] + hidden1_node1_activated * weights['hidden2_node0'][1]
hidden2_node0_activated = relu(hidden2_node0_output)
hidden2_node1_output = hidden1_node0_activated * weights['hidden2_node1'][0] + hidden1_node1_activated * weights['hidden2_node1'][1]
hidden2_node1_activated = relu(hidden2_node1_output)

# 출력층 계산
output = hidden2_node0_activated*weights['output'][0] + hidden2_node1_activated*weights['output'][1]
print("Forward propagation")
print(int(output))
print()
print()



### 역전파 ###
# 1. Update weight gradient for each layer. 가중치 기울기 업데이트
# 1-1. Output layer
target_value = 18
error = output - target_value
output_error_gradient = 2*error # MSE의 미분값. 예측값에 대한 오류의 기울기
output_weight_gradients = [
    hidden2_node0_activated*output_error_gradient,
    hidden2_node1_activated*output_error_gradient
]
print("Error back propagation")
print("Output layer:\t\t", output_weight_gradients)

# 1-2. Second hidden layer
hidden2_node0_error_gradient = output_error_gradient * weights['output'][0] * relu_derivative(hidden2_node0_activated)
hidden2_node1_error_gradient = output_error_gradient * weights['output'][1] * relu_derivative(hidden2_node1_activated)

hidden2_weight_gradients = {
    'hidden2_node0': [
        hidden2_node0_error_gradient * hidden1_node0_activated,
        hidden2_node0_error_gradient * hidden1_node1_activated
    ],
    'hidden2_node1': [
        hidden2_node1_error_gradient * hidden1_node0_activated,
        hidden2_node1_error_gradient * hidden1_node1_activated
    ]
}
print("Second hidden layer:\t", hidden2_weight_gradients)
print()

# 1-3. First hidden layer
hidden1_node0_error_gradient = (
    hidden2_node0_error_gradient * weights['hidden2_node0'][0] +
    hidden2_node1_error_gradient * weights['hidden2_node1'][0]
) * relu_derivative(hidden1_node0_activated)
hidden1_node1_error_gradient = (
    hidden2_node0_error_gradient * weights['hidden2_node0'][1] +
    hidden2_node1_error_gradient * weights['hidden2_node1'][1]
) * relu_derivative(hidden1_node1_activated)

hidden1_weight_gradients = {
    'hidden1_node0': [
        hidden1_node0_error_gradient * input_data[0],
        hidden1_node0_error_gradient * input_data[1]
    ],
    'hidden1_node1': [
        hidden1_node1_error_gradient * input_data[0],
        hidden1_node1_error_gradient * input_data[1]
    ]
}
print("First hidden layer:\t", hidden1_weight_gradients)
print()


# 2. Update the weight itself, using updated weight gradients and eta.
for key in weights.keys():
    if key.startswith('hidden1'):
        weights[key] = [w - learning_rate * g for w, g in zip(weights[key], hidden1_weight_gradients[key])]
    elif key.startswith('hidden2'):
        weights[key] = [w - learning_rate * g for w, g in zip(weights[key], hidden2_weight_gradients[key])]
    else:  # 출력층
        weights[key] = [w - learning_rate * g for w, g in zip(weights[key], output_weight_gradients)]

for key, value in weights.items():
    print(f"{key}: {value}")

'''zip 함수를 안썼을 때.'''
# for key in weights.keys() :
#     if key.startswith('hidden1') :
#         for i in range(len(weights[key])) :
#             weights[key][i] = weights[key][i] - learning_rate * hidden1_weight_gradients[key][i]
#     elif key.startswith('hidden2') :
#         for i in range(len(weights[key])) :
#             weights[key][i] = weights[key][i] - learning_rate * hidden2_weight_gradients[key][i]
#     else :
#         for i in range(len(weights[key])) :
#             weights[key][i] = weights[key][i] - learning_rate * output_weight_gradients[i]