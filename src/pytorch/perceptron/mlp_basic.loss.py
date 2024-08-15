import torch
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()

# Our dataset contains a single image of a dog, where
# cat = 0 and dog = 1 (corresponding to index 0 and 1)
target_tensor = torch.tensor([1])
print(target_tensor.item())

# Prediction: Most likely a dog (index 1 is higher)
# Note that the values do not need to sum to 1
predicted_tensor = torch.tensor([[2.0, 5.0]])
loss_value = loss_function(predicted_tensor, target_tensor)
print(loss_value.item())
# tensor(0.0181)

# Prediction: Slightly more likely a cat (index 0 is higher)
predicted_tensor = torch.tensor([[1.5, 1.1]])
loss_value = loss_function(predicted_tensor, target_tensor)
print(loss_value.item())

# Mean Square Errors MSE
# Define the loss function
loss_function = nn.MSELoss()

# Define the predicted and actual values as tensors
predicted_tensor = torch.tensor([320000.0])
actual_tensor = torch.tensor([300000.0])

# Compute the MSE loss
loss_value = loss_function(predicted_tensor, actual_tensor)
print(loss_value.item())  # Loss value: 20000 * 20000 / 1 = ...
# 400000000.0

'''
https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
https://pytorch.org/docs/stable/nn.html#loss-functions
'''
