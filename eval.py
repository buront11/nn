from neural_network import NeuralNetwork
import mnistDataLoader

import torch

test_data = mnistDataLoader.get_testloader()

model = NeuralNetwork()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: "+str(device))

model.to(device)

model.load_state_dict(torch.load("model/nn"))


total = 0.0
correct = 0.0

with torch.no_grad():
    for (inputs, labels) in test_data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))