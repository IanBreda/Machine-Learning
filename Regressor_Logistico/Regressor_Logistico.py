import random
from math import log10

def hwx(x, w):
    soma = 0
    for j in range(len(x)):
        soma += x[j] * w[j]
    result = 1/(1 + (2.718 ** (soma * (-1))))
    return result

def custo(x, w, y):
    soma = 0
    for i in range(len(x)):
        if y[i] == 1:
            soma -= log10(hwx(x[i], w))
        else:
            soma -= log10(1 - hwx(x[i], w))
    return soma/len(x)

def gradiente(x, w, y):
    learning_rate = 0.1
    aux = []
    for j in range(len(x[0])):
        soma = 0
        for i in range(len(x)):
            soma += (hwx(x[i], w) - y[i])*x[i][j]
        soma = soma * learning_rate / len(x)
        aux.append(soma)
    
    for i in range(len(w)):
        w[i] -= aux[i]

x = [[random.randint(-5, 5) for _ in range(3)] for _ in range(5)]
w = [random.randint(-5, 5) for _ in range(len(x[0]))]
y = [random.randint(0, 1) for _ in range(len(x))]

initial_cost = custo(x, w, y)
print('Valores de Y: ', y)
print('Custo Inicial:', initial_cost)
print('W inicial = ', w)
print('\nValores hw(x):')
for i in range(len(x)):
    print(hwx(x[i], w))

for epoch in range(10000):
    gradiente(x, w, y)
    cost = custo(x, w, y)
    if cost <= 0.01:
        print('Valores ideais de W encontrados na tentativa: ', epoch)
        break

final_cost = custo(x, w, y)
print('\nCusto Final:', final_cost)
print('W final = ', w)
print('\nValores finais hw(x):')
for i in range(len(x)):
    print(hwx(x[i], w))