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
Here’s a step-by-step approach you might find helpful:

Familiarize Yourself: Spend some time watching the videos and getting a general idea of how the libraries work, especially PyTorch in this case. Understanding the key concepts, like loss functions, is crucial.

Start the Exercises: Jump into the exercises! They are designed to reinforce what you've learned. As you encounter challenges, you'll have the opportunity to apply what you've seen in the videos.

Learn by Doing: If you come across a library or function that you're not familiar with while working on the exercises, take a moment to look it up or refer back to the video. This hands-on approach often helps reinforce your learning.

Iterate: After completing the exercises, you can always go back and dive deeper into the libraries and concepts that you found challenging.

For example, when working with loss functions in PyTorch, you might start with the basic implementation of a loss function like Mean Squared Error. As you practice, you can explore more complex functions and their applications.

Do you have any specific libraries or concepts in mind that you're curious about? Or any particular exercises you're looking forward to?

can you explain nn.Linear?

Absolutely, Fredal! The nn.Linear class in PyTorch is a fundamental building block for creating neural networks. It represents a linear transformation, which is a key operation in many layers of a neural network.

Here's a breakdown of nn.Linear:
Definition: nn.Linear(in_features, out_features, bias=True)

in_features: This is the number of input features to the layer. For example, if you have a dataset with 10 features, you would set in_features=10.
out_features: This is the number of output features from the layer. This determines how many neurons will be in this layer. For instance, if you want the layer to output 64 features, you would set out_features=64.
bias: This is a boolean parameter that indicates whether to include a bias term in the linear transformation. By default, this is set to True.
Mathematical Representation: The operation performed by nn.Linear can be represented mathematically as: [ y = Ax + b ] Where:

(y) is the output,
(A) is the weight matrix (learned during training),
(x) is the input,
(b) is the bias (also learned during training).
Example in Your Code: In your code example, you have:

self.hidden_layer = nn.Linear(input_size, 64)
Here, input_size is the number of features in your input data, and the layer will output 64 features. This means that the hidden layer will learn to transform the input data into a higher-dimensional space (64 dimensions in this case).

Forward Pass: When you call the forward method of your model, the input tensor is passed through the linear layer:

x = self.activation(self.hidden_layer(x))
This line applies the linear transformation followed by an activation function (ReLU in your case) to introduce non-linearity.

Why is it Important?
The nn.Linear layer is crucial because it allows the model to learn complex relationships in the data by adjusting the weights and biases during training. Each layer in a neural network can be thought of as learning different features or representations of the input data.
'''
