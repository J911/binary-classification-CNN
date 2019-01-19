# Jaemin Lee (aka, J911)
# 2019

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=10)
        self.conv2 = nn.Conv2d(30, 100, kernel_size=10)
        self.mp1 = nn.MaxPool2d(6)
        self.mp2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(105800, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 50)
        self.fc5 = nn.Linear(50, 10)
        self.fc6 = nn.Linear(10, 2)
    
    def forward(self, x):
        x = x.type('torch.FloatTensor') # torch.ByteTensor -> torch.FloatTensor
        in_size = x.size(0)

        x = self.mp1(x) # Dimension collapse
        x = F.relu(self.mp2(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x

