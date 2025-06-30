import torch

'''1. DERIVATIVES'''
# any function
def f(x) :
    return x**2
# derivative equation
def derivative(f, x, dx=1e-6) :
    return (f(x + dx) - f(x)) / dx
x=2
df_dx = derivative(f, x)
print(df_dx)
print()



'''2. PARTIAL DERIVATIVES'''
# any function
def f(x, y) :
    return x**2*y + y**3 + x*y
# derive, respect to x
def partial_derivative_x(f, x, y, dx=1e-6) :
    return (f(x+dx, y) - f(x, y)) / dx
# derive, respect to y
def partial_derivative_y(f, x, y, dy=1e-6) :
    return (f(x, y+dy) - f(x, y)) / dy
x = 1
y = 2
df_dx = partial_derivative_x(f, x, y)
df_dy = partial_derivative_y(f, x, y)
print(df_dx)
print(df_dy)
print()



'''3. AUTOGRAD'''
# make "requires_grad" of a tensor to TRUE
tensor1 = torch.tensor([1., 2., 3.])
print("tensor1 require autograd:", tensor1.requires_grad)
tensor2 = torch.tensor([1., 2., 3.], requires_grad=True)
print("tensor2 require autograd:", tensor2.requires_grad)
print()




'''4. CHECK DERIVED VALUE'''
# 이걸 True로 하면 torch는 해당 텐서에 대한 모든 연산을 추적함.
# 이거 없으면 .backward() call 못함. False 가 default.
x = torch.tensor(2.0, requires_grad=True)
f = x**2
f.backward()
print("x =", x.item(), "에서의 미분값:", x.grad.item())

# for multi-variable function
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
f = x**2*y + y**3 + x*y
f.backward()
# x = 1, y = 2 일때 x에 대한 편미분값
print(x.grad.item())
# x = 1, y = 2 일때 y에 대한 편미분값
print(y.grad.item())
print()



'''5. CONTROLLING AUTOGRAD'''
# autograd 사용 중지하고 모델 평가할때 등에 사용.

x = torch.tensor(1.0, requires_grad=True)
print(x.requires_grad) # True

# 그냥: x로 새 변수 y 만들었을 때.
y = x*2
print(y.requires_grad) # True

# torch.no_grad() 안에서 만들었을 때.
with torch.no_grad() :
    y = x*2
    print(y.requires_grad) # False

# detach(): make "False" version of a "True" tensor
z = x.detach()
print(z.requires_grad) # False

# controlling...
with torch.no_grad() :
    with torch.enable_grad() :
        y = x * 2
        print(y.requires_grad) # True