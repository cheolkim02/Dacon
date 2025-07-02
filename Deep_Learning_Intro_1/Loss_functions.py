import torch
import torch.nn as nn

'''1. MSE'''
y = torch.tensor([2.5, 4.0, 1.5], dtype=torch.float)
y_hat = torch.tensor([3.0, 4.0, 2.0], dtype=torch.float)

mse_loss = nn.MSELoss()
loss = mse_loss(y_hat, y)
print("1. MSE loss:", loss.item())
print()
print()



'''2. BINARY CROSS ENTROPY (BCE)'''
y_true = torch.tensor([0, 1, 0], dtype=torch.float)
y_pred = torch.tensor([0.2, 0.8, 0.0], dtype=torch.float)
y_pred_sigmoid = torch.sigmoid(y_pred)

bce_loss = nn.BCELoss()
loss = bce_loss(y_pred_sigmoid, y_true)
print("2. BCE loss:", loss.item())
print()
print()



'''3. CATEGORICAL CROSS ENTROPY (CCE)'''
# import torch.nn.functional as F
y_true = torch.tensor([1, 0, 2])
y_pred_logits = torch.tensor([[2.0, 1.0, 0.1],
                              [0.1, 1.5, 1.2],
                              [1.0, 0.2, 3.0]])
cross_entropy_loss = nn.CrossEntropyLoss()
loss = cross_entropy_loss(y_pred_logits, y_true)
print("3. Cross Entropy loss:", loss.item())