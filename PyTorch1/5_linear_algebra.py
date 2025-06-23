import torch

'''1. MULTIPLICATION'''
# 1-1 vector multiplication
vector1 = torch.tensor([1, 2, 3])
vector2 = torch.tensor([4, 5, 6])
dot_product = torch.dot(vector1, vector2)
print("dot product:", dot_product)
print()

# 1-2 matrix multiplication - 텐서의 "*"은 같은 위치 얘들끼리였음
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])
mul_matrix = torch.matmul(matrix1, matrix2)
print("matrix multiplication:\n", mul_matrix)
print()
print()



'''2. DETERMINANT'''
# 2-1 determinant - torch.linalg.det(matrix)
matrix_A = torch.tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=torch.float32)
det_A = torch.linalg.det(matrix_A)
print("determinant:", det_A)
print()
print()



'''3. INVERSE MATRIX'''
# 3-1 find inverse matrix - torch.linalg.inv
matrix_A = torch.tensor([[4.0, 7.0],
                         [2.0, 6.0]])
matrix_inv = torch.linalg.inv(matrix_A)
print("inverse matrix:\n", matrix_inv)
print()

# 3-2 check1 - A * inv(A) = I
print("check1:\n", torch.matmul(matrix_A, matrix_inv))
# 3-3 check2 - inv(A) * A = I
print("check2:\n", torch.matmul(matrix_inv, matrix_A))
print()

# 3-4 there is not inverse matrix if determinant = 0
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
# matrix_inv = torch.linalg.inv(matrix) ==> error!
try :
    torch.linalg.inv(matrix)
except Exception as e :
    print(e)
print()
print()



'''4. TRACE'''
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=torch.float32)
trace_A = torch.trace(A)
print("trace A:", trace_A)
print()
print()



'''5. EIGENVALUES AND EUGENVECTORS'''
A = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
eigenvalues, eigenvectors = torch.linalg.eig(A)
print("eigenvalues:\n", eigenvalues)
print("eigenvectors:\n", eigenvectors)
print()
print()



'''6. SVD, VALUE DECOMPOSITION(특이값 분해)'''
A = torch.tensor([[3.0, 2.0, 2.0],
                  [2.0, 3.0, -2.0]], dtype=torch.float32)
U, S, V = torch.linalg.svd(A, full_matrices=False)
print("matrix:\n", A)
print("U matrix:\n", U)
print("Singlar value diagonal:\n", torch.diag(S))
print("V^T matrix:\n", V.transpose(0, 1))
