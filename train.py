import mnistDataLoader

from neural_network import NeuralNetwork
from config import *

import torch


net = NeuralNetwork()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: "+str(device))

net.to(device)

criterion = net.get_criterion()
optimizer = net.get_optimizer()

train_data = mnistDataLoader.get_trainloader()

for epoch in range(epochs):
    running_loss = 0.0
    for index, data in enumerate(train_data):
        # まずは勾配をゼロに
        optimizer.zero_grad()

        inputs, labels = data[0].to(device), data[1].to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs)
        loss.backward()
        optimizer.step()
        
        running_loss = loss.item()
        if index % 100 ==99:
            print("epoch: &d, step: &d, loss: %3f" % (epoch+1,index+1,running_loss/100))
            
net = net.to('cpu')
torch.save(net.state_dict(), "model/nn")
print("Finished train")