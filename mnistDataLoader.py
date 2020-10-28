from config import *

import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import numpy as np

transoform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))]
)

trainset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transoform)
                                    
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=bacth_size,
                                            shuffle=True,
                                            num_workers=2)

testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transoform)

testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=bacth_size,
                                            shuffle=False,
                                            num_workers=2)

classes = tuple(np.linspace(0,9,10,dtype=np.uint8))

def get_trainloader():
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=2)
    return trainloader

def get_testloader():
    testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=100,
                                            shuffle=False,
                                            num_workers=2)
    return testloader