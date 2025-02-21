import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)

print(dataset.data.shape)