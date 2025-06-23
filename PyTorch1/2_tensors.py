import torch
import numpy as np

# 1. Creating tensor - tensor uses torch.Tensor class
# create tensor with a list
ls = [1, 2, 3, 4, 5]
tensor1 = torch.tensor(ls)
print(tensor1)

# create tensor with numpy
numpy_array = np.array([1.5, 2.5, 3.5])
tensor2 = torch.Tensor(numpy_array)
print(tensor2)

# 2. Data type
# integer type tensor
int_tensor = torch.tensor([1, 2, 3, 4])
print("int tensor values: ", int_tensor)
print("tensor type: ", int_tensor.dtype)
print()

# float type tensor
float_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("float tensor values: ", float_tensor)
print("tensor type: ", float_tensor.dtype)
print()

# boolean type tensor
bool_tensor = torch.tensor([True, False, True])
print("boolean tensor values: ", bool_tensor)
print("tensor type: ", bool_tensor.dtype)
print()

# specify type - "torch.---" int8~int64, uint8~uint64,
# float16(half), float32(float), float64(double)
int32_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
print("int32 tensor values: ", int32_tensor)
print("tensor type: ", int32_tensor.dtype)
print()

# changing data type of a tensor
original = torch.tensor([1, 2, 3, 4])
float_tensor = original.float()
int_tensor = float_tensor.int()
double_tensor = int_tensor.double()
bool_tensor = int_tensor.bool()

print("처음 생성한 텐서:", original.dtype)
print('float_tensor의 타입 확인: ', float_tensor.dtype)
print('int_tensor의 타입 확인: ', int_tensor.dtype)
print('double_tensor의 타입 확인: ', double_tensor.dtype)
print('bool_tensor의 타입 확인: ', bool_tensor.dtype)


# dimension and shape of tensor
data = [1, 2, 3]
vector = torch.tensor(data)
print("vector: ", vector)
print("dimension: ", vector.dim()) # dimension: 1
print("shape: ", vector.shape) # shape: torch.Size([3])

data = [[1, 2, 3],
        [4, 5, 6]]
matrix = torch.tensor(data)
print("matrix: ", matrix)
print("dimension: ", matrix.dim()) #dimension: 2
print("shape: ", matrix.shape) # shape: torch.Size([2, 3])

data = [
    [[1, 2, 3],
     [3, 4, 5]],
    [[5, 6, 7],
     [7, 8, 9]]
]
tensor3D = torch.tensor(data)
print("tensor3D: ", tensor3D)
print("diemention: ", tensor3D.dim()) # dimension: 3
print("shape: ", tensor3D.shape) # shape: torch.Size([2, 2, 3])
print()
print()

# quick tensor create
zero_tensor = torch.zeros(2, 3)
print("zero tensor: \n", zero_tensor)

one_tensor = torch.ones(2, 3)
print("one tensor: \n", one_tensor)

full_tensor = torch.full((2, 3), 5)
print("full tensor - 5: \n", full_tensor)

eye_tensor = torch.eye(3, 3)
print("eye tensor: \n", eye_tensor)

# [0,10) 2씩 증가
arange_tensor = torch.arange(0, 10, 2)
print("arange tensor: \n", arange_tensor)
# [0,16) 1씩 증가. (0~15)
arange_tensor2 = torch.arange(16)
print("arange tensor 2: \n", arange_tensor2)

linspace_tensor = torch.linspace(0, 10, 5)
print("linspace tensor: \n", linspace_tensor)
# arange and linspace tensors are always 1 dimension
print()
print()


# random
# rand(): uniform dist. between 0 and 1
rand_tensor = torch.rand(2, 3)
print("rand tensor: \n", rand_tensor)

# randn(): uniform dist. ~N(0, 1)
randn_tensor = torch.randn(2, 3)
print("randn tensor: \n", randn_tensor)

# randint(): in range
randint_tensor = torch.randint(low=0, high=10, size=(2, 3))
print("randint tensor: \n", randint_tensor)

# randperm(): random order from 0 to n-1
randperm_tensor = torch.randperm(10)
print("randperm tensor: \n", randperm_tensor)
print()
print()


answer = torch.randint(low=0, high=10, size=(3, 4), dtype=torch.int32)
print(answer)