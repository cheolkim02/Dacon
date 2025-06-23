import torch
''' 1. RESHAPING TENSOR '''
'''1-1 reshape and view example'''
x = torch.arange(16)
print("tensor x: ", x)
print("x shape: ", x.shape)
print()

# reshape - 딱 맞는 shape으로만 reshape 가능
reshaped_x = x.reshape(4, 4)
print("reshaped x: \n", reshaped_x)
print("reshaped x shape: ", reshaped_x.shape)
print()

# view - x를 다른 차원으로 보기 - 메모리상 연속 필수
viewed_x = x.view(4, 2, 2)
print("viewed x: \n", viewed_x)
print("viewed x shape: ", viewed_x.shape)
print()
print()

'''1-2 sqeeze and unsqueeze example'''
y = torch.tensor([[[1, 2, 3, 4]]]) # 크기: (1, 1, 4)
print("tensor y: ", y)
print("y shape: ", y.shape)
print()

# sqeeze
squeezed_y = y.squeeze()
print("sqeezed y: ", squeezed_y)
print("squeezed y shape: ", squeezed_y.shape)
print()
squeezed_dim_y = y.squeeze(0) # squeeze certain dimension only
print("squeezed dim y: ", squeezed_dim_y)
print("squeezed dim y shape: ", squeezed_dim_y.shape)
print()

# unsqeeze
z = torch.tensor([1, 2, 3, 4])
print("tensor z: ", z)
print("z shape: ", z.shape)
print()
unsqueeze_first_dim_z = z.unsqueeze(0)
print("unsqueeze_first_dim_z: ", unsqueeze_first_dim_z)
print("unsqueeze_first_dim_z shape: ", unsqueeze_first_dim_z.shape)
print()
unsqueeze_last_dim_z = z.unsqueeze(-1)
print("unsqueeze_last_dim_z: \n", unsqueeze_last_dim_z)
print("unsqueeze_last_dim_z shape: ", unsqueeze_last_dim_z.shape)
print()
print()

'''1-3 transpose example'''
tensor = torch.rand(2, 3, 4) # rand 0~1, dim: (2, 3, 4)
print("tensor: \n", tensor)

# transpose로 1번과 2번 차원을 교환
transposed_tensor = tensor.transpose(0, 1)
print("transposed tensor: \n", transposed_tensor) # (3, 2, 4)
print()

# permute로 차원 순서 재배열
permuted_tensor = tensor.permute(2, 0, 1)
print("permuted tensor: \n", permuted_tensor)
print()
print()



''' 2. JOINING TENSOR'''
'''2-1 joining tensor'''
# cat: (2x2) + (2x3) = (2x5)
tensor1 = torch.tensor([[1, 2], 
                        [3, 4]])
print("tensor1: \n", tensor1)
tensor2 = torch.tensor([[5, 6, 7], 
                        [8, 9, 0]])
print("tensor2: \n", tensor2)
# 1번 차원(열)을 따라 결합
cat_tensor = torch.cat((tensor1, tensor2), dim=1)
print("cat tensor: \n", cat_tensor)
print()

# stack: (2x2) + (2x2) + (2x2) = (3x2x2)
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
tensor3 = torch.tensor([[9, 10], [11, 12]])
# 새 차원을 0번 차원으로 삽입하여 쌓기
stack_tensor = torch.stack((tensor1, tensor2, tensor3), dim=0)
print("stack tensor: \n", stack_tensor)
print()

'''2-2 splitting tensor'''
tensor = torch.arange(10)
print("original: ", tensor)
# chunk: tensor를 몇 개의 조각으로 나눌 것인가 + 어느 차원 기준
# 텐서를 3개로 나누기. 크기는 자동.
chunk_tensor = torch.chunk(tensor, 3, dim=0)
print("chunk tensor: \n", chunk_tensor)
for i, chunk in enumerate(chunk_tensor) :
    print("Chunk", i, ":", chunk, chunk.shape)
print()

# split: chunk 상위호환. 몇 개씩 배분할 것인가
split_tensor = torch.split(tensor, (3, 3, 3, 1), dim=0)
print("split tensor: \n", split_tensor)
for i, split in enumerate(split_tensor) :
    print("Split", i, ":", split)
print()
print()



'''3. SPLITTING TENSOR'''
# tensor indexing
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
selected_element = tensor[1, 2] # tensor(6)
print("selected element: ", selected_element)
print()

# tensor slicing
tensor = torch.arange(0, 25).reshape(5, 5)
print("original: \n", tensor)
sliced_tensor = tensor[:, 1:4] # all rows, columns 1, 2, 3
print("sliced tensor: \n", sliced_tensor)

# conditional indexing
tensor = torch.arange(0, 25).reshape(5, 5)
mask = tensor>5 # make a boolean condition
selected_elements = tensor[mask]
print("elements >5\n", selected_elements)
print()
print()



'''4. ADVANCED INDEXING'''
# gather
y = torch.tensor([[1, 2],
                  [3, 4]])
# 1번행에서 0번째, 0번째. 2번행에서 1번째, 0번째. 2x2 텐서 반환
gather_index = torch.tensor([[0, 0], [1, 0]])
gathered = y.gather(1, gather_index)
print("gathered: \n", gathered)
print()

# masked select: 조건부 선택
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8]])
mask = torch.tensor([True, False, True, False])
# 0번째, 2번째 값 반환
selected = tensor.masked_select(mask)
print("masked_selected: \n", selected)
print()

# integer array indexing
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 2])
# (0, 1)값이랑 (2, 2)값을 1차원 텐서로 반환
selected_by_index = tensor[rows, cols]
print("selected by index: \n", selected_by_index)
print()
print()



import numpy as np
'''5. BETWEEN TENSOR AND NUMPY'''
# torch.from_numpy - transform numpy array to into tensor
numpy_array = np.array([1, 2, 3, 4, 5])
print("numpy array: ", numpy_array)
tensor_from_numpy = torch.from_numpy(numpy_array)
print("tensor from numpy: ", tensor_from_numpy)
print()

# tensor.numpy - transform tensor into numpy array
a = torch.tensor([1, 2, 3, 4, 5])
print("tensor a: ", a)
numpy_from_tensor = a.numpy()
print("numpy from tensor: ", numpy_from_tensor)
print()

# 얘네는 메모리 공유하는 함수임. 변환 후 값 바꾸면 다른 얘도 값 바뀜

# clone - cloning tensor
original_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
cloned_tensor = original_tensor.clone()
cloned_tensor[0] = 10
print("original: ", original_tensor)
print("clone: ", cloned_tensor)