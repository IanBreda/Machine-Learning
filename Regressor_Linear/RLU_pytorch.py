import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Dados de entrada X e saída Y com w = 5 e b = -1 
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
y = torch.tensor([[4.0], [9.0], [14.0], [19.0], [24.0], [29.0], [34.0], [39.0], [44.0], [49.0]])

# Definição do modelo de regressão linear usando PyTorch
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_size, output_size)  # Camada linear

    def forward(self, x):
        return self.lin(x)  # Forward pass

# Inicialização do modelo, taxa de aprendizado e otimizador
input_size, output_size = 1, 1
model = LinearRegression(input_size, output_size)
lr = 0.1  # Taxa de aprendizado
max_epochs = 10000  # Máximo de épocas
loss = nn.MSELoss()  # Função de perda (erro quadrático médio)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Otimizador Adam
tol = 1e-5  # Tolerância para parar o treinamento

# Listas para armazenar evolução do erro, peso e bias
evo_erro = []
evo_peso = []
evo_bias = []

# Loop de treinamento
for e in range(max_epochs):
    pred = model(x)  # Predição do modelo
    l = loss(y, pred)  # Cálculo do erro
    evo_erro.append(float(l))  # Armazena o erro atual
    # Parada antecipada se o erro for suficientemente pequeno
    if l <= tol:
        break

    # Armazena peso e bias atuais (sem gradientes)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                evo_peso.append(param.item())
            elif 'bias' in name:
                evo_bias.append(param.item())

    l.backward()  # Cálculo do gradiente
    optimizer.step()  # Atualização do peso e bias
    optimizer.zero_grad()  # Reseta os gradientes

    # Exibe os parâmetros a cada 50 épocas
    if e % 50 == 0:
        print(f'\nIteração {e}:')
        for name, param in model.named_parameters():
            print(f'{name} = {param.item()}')

# Teste final com entrada x = 20
print('\n\nTeste com resultados finais de pesos e bias')
xtest = torch.tensor([[20.0]])
print(model(xtest))  # Saída prevista

# Gráficos lado a lado: erro vs peso/bias
plt.figure(figsize=(12, 5))

# Gráfico 1 - Evolução do erro
plt.subplot(1, 2, 1)
plt.plot(evo_erro, label='Erro (Loss)', color='blue')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.title('Evolução do Erro')
plt.grid(True)
plt.legend()

# Gráfico 2 - Evolução do peso e do bias
plt.subplot(1, 2, 2)
plt.plot(evo_peso, label='Peso', color='green')
plt.plot(evo_bias, label='Bias', color='red')
plt.xlabel('Épocas')
plt.ylabel('Valor')
plt.title('Evolução do Peso e Bias')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Gráfico final - Comparação entre dados reais e reta aprendida
x_np = x.cpu().detach().numpy()
y_np = y.cpu().detach().numpy()
pred_np = model(x).cpu().detach().numpy()

plt.scatter(x_np, y_np, label='Dados reais')  # Pontos reais
plt.plot(x_np, pred_np, color='red', label='Reta aprendida')  # Linha da regressão
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regressão Linear')
plt.grid()
plt.show()