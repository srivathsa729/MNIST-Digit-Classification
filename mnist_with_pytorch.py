import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np

from six.moves import urllib    
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

train = datasets.MNIST("", train = True, download = True, transform = transforms.ToTensor())

test = datasets.MNIST("", train = False, download = True, transform = transforms.ToTensor())



trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

for data in trainset:
  print(data)
  break

import matplotlib.pyplot as plt

plt.imshow(data[0][0].view([28, 28]))
plt.show()

print(data[1][0])

total = 0
counter = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
  Xs, ys = data
  for yi in ys:
    counter[int(yi)] += 1
    total += 1

print(counter)

for i in counter:
  print(f"{i}:{(counter[i]/total)*100} ")

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28*28, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(64, 10)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)

    return F.log_softmax(x, dim = 1)


net = Net()
print(net)

temp = torch.rand((28, 28))

plt.imshow(temp)
plt.show()

temp = temp.view(-1, 28*28)
temp.shape

output = net(temp)

print(output)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.0001)

EPOCHS = 3

for epoch in range(EPOCHS):
  for data in trainset:
    X, y = data
    net.zero_grad()
    output = net(X.view(-1, 28*28))
    loss = F.nll_loss(output, y)
    loss.backward()
    optimizer.step()
  print(loss)

correct = 0
total = 0
with torch.no_grad():
  for data in trainset:
    X, y = data
    output = net(X.view(-1, 28*28))
    for idx, i in enumerate(output):
      if torch.argmax(i) == y[idx]:
        correct += 1
      total += 1

print(correct/total)

