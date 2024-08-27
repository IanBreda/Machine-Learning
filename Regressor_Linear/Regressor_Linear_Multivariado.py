# Criado por Ian Breda, 2024

#Usando apenas a biblioteca random
import random

#A função de custo, recebe os valores de x, y, w e a taxa de regularização
def calcula_custo(x, y, w, adjust_rate):
    soma_total = 0

    #Percorre primeiro todo o hw(x)
    for i in range(len(x)):
        aux = 0
        for j in range(len(x[0])):
           aux += x[i][j] * w[j]
        aux -= y[i]
        aux = aux ** 2
        soma_total += aux
    
    #Por fim soma todo peso w^2 com a regularização
    #E então divide pelo numero de amostras no conjunto X
    for j in range(len(x[0])):
        soma_total += adjust_rate * (w[j] ** 2)

    soma_total = soma_total/(2 * len(x))

    return soma_total

#Na função de cálculo das derivadas, recebe todos os parâmetros do programa
def calcula_derivadas(x, y, w, learning_rate, adjust_rate):
    #Cria um auxiliar que vai receber os valores atualizados de W, antes que eles sejam realmente atualizados
    aux_w = [0 for _ in range(len(w))]

    #Calcula uma derivada parcial, dependendo se é o W0 ou não
    for j in range(len(x[0])):
        soma = 0
        if j == 0:
            for i in range(len(x)):
                aux = 0
                for k in range(len(x[0])):
                    aux += x[i][k] * w[k]
                aux -= y[i]
                soma += aux
            soma = soma * learning_rate / len(x)

        else:
            for i in range(len(x)):
                aux = 0
                for k in range(len(x[0])):
                    aux += x[i][k] * w[k]
                aux -= y[i]
                aux = aux * x[i][j]
                soma += aux
            soma = soma / len(x)
            soma = soma + (adjust_rate * w[j] / len(x))
            soma = soma * learning_rate
        
        aux_w[j] = soma
    
    #Por fim atualiza todos os valores de W
    for i in range(len(w)):
        w[i] -= aux_w[i]
    
    return w

#Inicia o programa gerando valores aleatórios para X e W
#O Y é gerado com base em X. Nesse programa em específico, o Y é gerado pegando cada valor de X,
#E multiplicando pelo tamanho do proprio conjunto de X (de forma crescente, onde o primeiro elemento é multiplicado por 1, o segundo por 2...). 
# Portanto, suponha que uma das amostras de X seja: [1, 1, 1, 1, 1]
#Logo, ao ser multiplicado, X se torna: [1, 2, 3, 4, 5]. E como Y é a soma de todo o X, Y será igual a 1 + 2 + 3 + 4 + 5 = 15
#Para 'TODOS' os elementos de X

#X é criado de forma que todo x0 é 1, seguido de mais 4 elementos, para todas as 6 amostras
x = [[1] + [random.randint(1, 3) for _ in range(4)] for _ in range(6)]
y = []
for a in range(len(x)):
    somaY = 0
    for f in range(len(x[0])):
        somaY += x[a][f]*(f+1)
    y.append(somaY)

#Como dito anteriormente, W é gerado aleatoriamente. Mas no fim, deve ser os valores que multiplicam X (os pesos).
#Neste exemplo de código, W será [1, 2, 3, 4, 5], ou algo próximo disso.
w = [random.randint(1, 5) for _ in range(len(x[0]))]

#A taxa de aprendizado e de regularização são escolhidas.
#O programa funciona melhor quando a learning rate é baixa e regularização é zerada.
learning_rate = 0.08
adjust_rate = 0

#Aqui foi definido critérios de parada (10000 iterações ou um erro < que 10^-6)
max_iteracoes = 10000
tolerancia = 1e-6

for iteracao in range(max_iteracoes):
    custo = calcula_custo(x, y, w, adjust_rate)
    if abs(custo) < tolerancia:
        print(f'Convergência atingida na iteração {iteracao + 1}')
        break
    w = calcula_derivadas(x, y, w, learning_rate, adjust_rate)

print(f'Custo final: {custo}')
print(f'Pesos finais: {w}')