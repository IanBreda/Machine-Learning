import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

batchsize = 50
learning = 0.01
max_epochs = 5
camadas = [784, 30, 30, 10, 10]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

# Criando DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self, camadas):
        super(NeuralNetwork, self).__init__()
        camada = []
        for i in range(len(camadas)-1):
            camada.append(nn.Linear(camadas[i], camadas[i+1]))
            if i < len(camadas)-2:
                camada.append(nn.ReLU())
        self.net = nn.Sequential(*camada)

    
    def forward(self, x):
        return self.net(x)

modelo = NeuralNetwork(camadas)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelo.parameters(), lr=learning)

total = len(train_loader)

for e in range(max_epochs):
    print('\n')
    for i, (img, label) in enumerate(train_loader):
        imagem = img.reshape(-1, 784)

        pred = modelo(imagem)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f'Epoca: {e+1} concluido {i}/{total} loss = {loss.item():.4f}')

with torch.no_grad():
    correto = 0
    total = len(test_loader.dataset)
    imagens_plotadas = 0

    for image, label in test_loader:
        image_flat = image.reshape(-1, 784)

        output = modelo(image_flat)
        _, pred = torch.max(output, 1)
        correto += (pred == label).sum().item()

        # Mostrar só as 9 primeiras imagens do dataset
        if imagens_plotadas < 25:
            for j in range(image.size(0)):
                if imagens_plotadas >= 25:
                    break
                plt.subplot(5, 5, imagens_plotadas + 1)
                plt.imshow(image[j][0], cmap='gray')
                plt.title(f'Pred: {pred[j].item()}, {label[j].item()}')
                plt.axis('off')
                imagens_plotadas += 1
    
    acc = correto / total
    print(f'\nO modelo obteve {correto}/{total} amostras corretas, acuracia = {acc:.2f}')

plt.tight_layout()
plt.show()