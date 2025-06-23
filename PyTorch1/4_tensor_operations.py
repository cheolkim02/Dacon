import torch

'''1. BASIC OPERATIONS'''
# 1-1 operating tensor and scalar - 모든 칸에 각각 적용
tensor = torch.tensor([[1, 2],
                       [3, 4]], dtype=torch.float32)
add_scalar = tensor + 1
sub_scalar = tensor - 1
mul_scalar = tensor * 2
div_scalar = tensor / 2
pow_scalar = tensor ** 2
print("add scalar: \n", add_scalar)
print("sub scalar: \n", sub_scalar)
print("mul scalar: \n", mul_scalar)
print("div scalar: \n", div_scalar)
print("pow scalar: \n", pow_scalar)
print()

# 1-2 add/sub between tensors - 동일 위치 얘들끼리
a = torch.tensor([[1, 2], 
                  [3, 4]])
b = torch.tensor([[5, 6], 
                  [7, 8]])
add_tensors = a+b
sub_tensors = a-b
print("add_tensors: \n", add_tensors)
print("sub_tensors: \n", sub_tensors)
print()

# 1-3 mul/div between tensors - 동일 위치 얘들끼리
a = torch.tensor([2, 3, 4])
b = torch.tensor([3, 4, 5])
mul_tensors = a*b
div_tensors = a/b
print("mul_tensors: \n", mul_tensors)
print("div_tensors: \n", div_tensors)
print()

# 1-4 broadcasting - operate between tensors of differing shapes
# one of smaller size is enlarged to match the bigger one
# add dimension 1 (however many needed) at the front.
# (1, 3)*(3) -> (1, 3)*(1, 3)
a = torch.tensor([[1, 2, 3]], dtype=torch.float32) # size(1, 3)
b = torch.tensor([4, 5, 6], dtype=torch.float32) # size(3)
result = a + b
print("add by bradcast: \n", result)
print()



'''2. COMPARITON OPERATORS'''
# returns a boolean tensor
# 2-1 comparison operations
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([1, 2, 0, 4, 5])
result = torch.eq(a, b)
print("equivalence result: ", result)

a = torch.tensor([5, 6, 7, 8])
b = torch.tensor([4, 6, 7, 10])
gt_result = a > b # torch.gt(a, b)
ge_result = a >= b # torch.ge(a, b)
lt_result = a < b # torch.lt(a, b)
le_result = a <= b # torch.le(a, b)
print("a > b", gt_result)
print("a >= b", ge_result)
print("a < b", lt_result)
print("a <= b", le_result)

# 2-2 condition operations
a = torch.tensor([1, 4, 3, 2, 5])
b = torch.tensor([3, 2, 1, 5, 4])
mask = torch.gt(a, b)
# mask = tensor([False, True, True, False, True])
selected_elements = a[mask] # 숫자랑 불리언임. True 해당만 출력
print("selected elements: ", selected_elements)
print()
print()



'''3. max, min, sum, mean, median, mode'''
# 3-1 max/min
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
# returns biggest value of the tensor
max_value = torch.max(a)
print("max: ", max_value)
# returns biggest values and its indices for each of dim=1(cols)
max_values, max_indicies = torch.max(a, dim=1)
print("max values:", max_values) # larges values of each column
print("indices:", max_indicies) # indices of each largest value
print()

# 3-2 sum
total_sum = torch.sum(a) # total sum, integer
col_sum = torch.sum(a, dim=0) # sum of each col, tensor
row_sum = torch.sum(a, dim=1) # sum of each row, tensor
print("col sum:", col_sum)
print("row_sum:", row_sum)
print()

# 3-3 mean, median, mode
a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float)
mean_val = torch.mean(a)
median_val = torch.median(a)
mode_val, mode_idx = torch.mode(a)
print("mean:", mean_val)
print("median:", median_val)
print("mode, mode idx:", mode_val, mode_idx)
print()

# 3-4 variation, standard deviation
a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
variance = torch.var(a)
standard_deviation = torch.std(a)
print("variance:", variance)
print("standard deviation:", standard_deviation)
print()
print()



'''4. LOGICAL OPERATIONS'''
# 4-1 and, or, not - 모두 동일 위치에 있는 얘들끼리 연산
a = torch.tensor([True, False, True, False],dtype=torch.bool)
b = torch.tensor([True, True, False, False],dtype=torch.bool)
result_and = a&b # torch.Logical_and(a, b)
result_or = a|b # torch.Logical_or(a, b)
result_not = ~a # torch.Logical_not(a)
print(result_and)
print(result_or)
print(result_not)
print()
print()



'''5. MATH OPERATIONS'''
# 5-1 exp, log, sqrt
a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
exp_result = torch.exp(a)
log_result = torch.log(a)
sqrt_result = torch.sqrt(a)
print("exp:", exp_result)
print("log:", log_result)
print("sqrt:", sqrt_result)

# 5-2 pow, abs, reciprocal(역수), neg
a = torch.tensor([1, -2, 3, -4], dtype=torch.float32)
cubed_values = torch.pow(a, 3)
abs_values = torch.abs(a)
reciprocal_values = torch.reciprocal(a)
neg_values = torch.neg(a)
print("cubed:", cubed_values)
print("abs:", abs_values)
print("reciprocal:", reciprocal_values)
print("neg values:", neg_values)
print()

# 5-3 trigonometry - sin cos tan
angles = torch.tensor([torch.pi/6, torch.pi/4,
                       torch.pi/3, torch.pi/2, 0])
sin_values = torch.sin(angles)
cos_values = torch.cos(angles)
tan_values = torch.tan(angles)
print("sin values:", sin_values)
print("cos values:", cos_values)
print("tan values:", tan_values)

