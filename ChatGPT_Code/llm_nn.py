# Helpful video: https://www.youtube.com/watch?v=YDiSFS-yHwk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate training data
x = torch.unsqueeze(torch.linspace(-6, 6, 100), dim=1)
y = torch.sin(x) - 0.1 * x**2

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 4),
            nn.Sigmoid(),
        )
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x

net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training the network
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Generate predictions
predicted = net(x).detach()

# Plot the results
plt.figure()
plt.plot(x, y, 'r', label='True')
plt.plot(x, predicted, 'b', label='Predicted')
plt.legend()
plt.show()
