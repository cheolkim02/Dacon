
'''순전파만 사용하여 집 가격 예측 모델 구현'''


'''1. ACTIVATION FUNCTION: LINEAR'''
print()

input_data = [30, 2]
weights = {
    'node_0': [50, 10],
    'node_1': [-1, 0],
    'output': [0.5, 25]
}

node_0_value = input_data[0]*weights['node_0'][0] + input_data[1]*weights['node_0'][1]
node_1_value = input_data[0]*weights['node_1'][0] + input_data[1]*weights['node_1'][1]
hidden_layer_values = [node_0_value, node_1_value]
print("hidden layer values:", hidden_layer_values)

output = hidden_layer_values[0]*weights['output'][0] + hidden_layer_values[1]*weights['output'][1]
print("final cost estimation:", output)
print()
print()



'''2. ACTIVATION FUNCTION: RELU'''
def relu(x) :
    return max(0, x)

input_data = [30.0, 2.0]
weights = {
    'hidden1_node0': [50.0, 10.0],
    'hidden1_node1': [-1.0, 0.0],
    'hidden2_node0': [0.1, 10.0],
    'hidden2_node1': [-0.1, 0.0],
    'output': [0.1, 25.0]
}

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

# Output layer
output = hidden2_node0_activated * weights['output'][0] + hidden2_node1_activated * weights['output'][1]
print("final cost estimation:", int(output))
print()
print()



''''''