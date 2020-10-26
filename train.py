import mnistDataLoader

from neural_network import NeuralNetwork
from config import *


net = NeuralNetwork()

criterion = net.get_criterion()
optimizer = net.get_optimizer()

train_data = mnistDataLoader.get_trainloader()

for epoch in range(epochs):
    for index, data in enumerate(train_data):
        # まずは勾配をゼロに
        optimizer.zero_grad()
