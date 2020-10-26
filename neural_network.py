from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3) #28x28 -> 26x26
        self.conv2 = nn.Conv2d(32,64,3) #26x26 -> 24x24
        self.fc1 = nn.Linear(24*24*64,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,24*24*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_criterion():
        return nn.CrossEntropyLoss()

    def get_optimizer(self,lr=0.001,momentum=0.9):
        return optim.SGD(self.parameters(), lr=lr, momentum=momentum)