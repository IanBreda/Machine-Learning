import numpy as np
import matplotlib.pyplot as plt
P_CROSS = 0.6
P_MUT = 0.02

def gerar_individuos(tam_alvo):
    return np.random.randint(0, 2, tam_alvo)

def aptidao_individuo(alvo, ind):
    return np.sum(alvo == ind)

def roleta(qtd_ind, apt):
    return np.random.choice(range(qtd_ind), p=apt, size=qtd_ind)

def crossover(individuos, selecao, ponto, cross):
    indice = 0
    n_ind = []
    for p in cross:
        pai_a = individuos[selecao[indice]]
        pai_b = individuos[selecao[indice+1]]
        filho_a = []
        filho_b = []
        if p > P_CROSS:
            filho_a = np.concatenate((pai_a[:ponto], pai_b[ponto:]))
            filho_b = np.concatenate((pai_b[:ponto], pai_a[ponto:]))
        else:
            filho_a = pai_a.copy()
            filho_b = pai_b.copy()
        n_ind.append(filho_a)
        n_ind.append(filho_b)
        indice += 2

    return n_ind

def mutacao(new_individuos, tam_alvo, qtd_ind):
    for j in range(qtd_ind):
        for i in range(tam_alvo):
            p = np.random.uniform(0, 1)
            if p <= P_MUT:
                new_individuos[j][i] = 1 if new_individuos[j][i] == 0 else 0
    return new_individuos


#Cria o alvo
tam_alvo = int(input('Selecione o tamanho do alvo: '))
alvo = np.random.randint(0, 2, tam_alvo)

#Seleciona a quantidade e gera os indivíduos
individuos = []
qtd_ind = int(input('Selecione a quantidade de individuos a serem gerados (numero par): '))
for i in range(qtd_ind):
    individuos.append(gerar_individuos(tam_alvo))

geracoes = 5000
evo_acerto = []
for g in range(geracoes):
    
    #Cria vetor com a aptidao de cada individuo, usando a distancia de hamming
    apt = []
    apt = [aptidao_individuo(alvo, ind) for ind in individuos]

    #Sorteia os individuos por roleta
    #Normaliza aptidão
    soma_aptidao = sum(apt)
    apt = [x/soma_aptidao for x in apt]
    selecao = roleta(qtd_ind, apt)

    #Sorteio da prob. de cross-over e geração de novos individuos
    cross = np.random.uniform(0, 1, int(qtd_ind/2))
    ponto = np.random.randint(1, tam_alvo-1)
    individuos = crossover(individuos, selecao, ponto, cross)

    #Mutação
    individuos = mutacao(individuos, tam_alvo, qtd_ind)

    #Recalcula a aptidão após cross-over e mutação
    apt = [aptidao_individuo(alvo, ind) for ind in individuos]
    melhor_apt = max(apt)
    media_apt = sum(apt) / qtd_ind

    evo_acerto.append(melhor_apt)
    #Verifica se foi encontrado a solução
    print(f'Geração {g+1}: Melhor aptidão = {melhor_apt}, Média = {media_apt:.2f}')
    if melhor_apt == tam_alvo:
        print(f"Alvo atingido na geração {g+1}!")
        break

# Exibir resultado final
final_apt = [aptidao_individuo(alvo, ind) for ind in individuos]
melhor_idx = np.argmax(final_apt)
print(f'\nMelhor indivíduo encontrado: {individuos[melhor_idx]}')
print(f'Alvo:                        {alvo}')
print(f'Aptidão final: {final_apt[melhor_idx]}/{tam_alvo}')

plt.scatter(range(len(evo_acerto)), evo_acerto, color='red', label='Melhor Aptidão')
plt.plot(range(len(evo_acerto)), evo_acerto, color='blue', label='Melhor Aptidão')
plt.show()