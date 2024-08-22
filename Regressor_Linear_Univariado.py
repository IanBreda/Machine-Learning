# Criado por Ian Breda, 2024

#Importando as bibliotecas importantes
import random
import matplotlib.pyplot as plt

#Recebe os valores corretos de w0 e w1,
#Os valores de X (entrada) e Y (saida)
#E tbm um histórico de todos os custos da função custo
def plotagem(custos_totais, x, y, w0, w1):
    #Função de plotagem de gráficos
    y_pred = [w0 + w1 * xi for xi in x]
    plt.scatter(x, y, color='blue')
    plt.plot(x, y_pred, color='red')
    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')
    plt.title('Linear Regression')
    plt.show()
    #Faz uma plotagem do gráfico, onde imprime pontos azuis para os valores reais
    #E imprime tbm uma linha vermelha para o 'predict' da máquina (a reta deve passar por cima dos pontos!)

    qtd_custos = [i for i in range(len(custos_totais))]
    plt.scatter(qtd_custos, custos_totais)
    plt.show()
    #Por fim, imprime um gráfico que mostra um histórico da função custo


#A função do regressor linear recebe os valores do conjunto X e Y
#Recebe os valores de w0 e w1 (gerados aleatoriamente) que devem ser atualizados
#Recebe o tamanho (tam) do vetor X e Y (devem ter os mesmos tamanhos)
#Uma learning rate adequada (muito pequeno demora para convergir e muito grande gera problemas)
#Uma tolerância de erro pequena, um limite máximo de iterações e um contador de iterações
def linear_regressor(x, y, w0, w1, tam, learning = 0.13, tol = 0.00000001, max = 900, iterator = 1):
    #Função para atualizar os valores de w0 e w1
    #Caso ja tenha atingido o limite de iterações, encerra o algoritmo
    if iterator > max:
        print('Limite de iterações atingido')
        return [w0, w1]
    
    #Calculo da função custo
    soma_custo = 0
    for i in range(tam):
        soma_custo += ((w0 + w1*x[i]) - y[i])**2
    Jw = soma_custo/(2*tam)

    print(f'iteração {iterator}: W0 = {w0}  W1 = {w1}  Custo = {Jw}')
    custos_totais.append(Jw)
    
    #Se o custo for menor ou igual a tolerância, encerra o algoritmo
    if abs(Jw) <= tol:
        return [w0, w1]
    
    #Atualização dos valores de w0 e w1
    soma_w0 = 0
    soma_w1 = 0
    for j in range(tam):
        soma_w0 += ((w0 + w1*x[j]) - y[j])
        soma_w1 += ((w0 + w1*x[j]) - y[j])*x[j]
    w0 -= soma_w0*(learning/tam)
    w1 -= soma_w1*(learning/tam)

    #Chama a função recursivamente
    return linear_regressor(x, y, w0, w1, tam, learning, tol, max, iterator + 1)

#Criação inicial dos valores: X e Y são escolhidos e w0 e w1 são randomizados (entre 1 e 100)
x = [1, 2, 3, 4]
y = [3, 5, 7, 9]
w0 = random.randint(1, 100)
w1 = random.randint(1, 100)
tam = len(x)
custos_totais = []

#Imprime os melhores valores do conjunto W e chama a função de plotagem
print('Procurando o melhor valor para W[w0, w1]')
vetorW = []
vetorW = linear_regressor(x, y, w0, w1, tam)

print(f'\n melhores valores de W = {vetorW}')
plotagem(custos_totais, x, y, vetorW[0], vetorW[1])