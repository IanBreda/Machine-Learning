import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev

P_CROSS = 0.50
P_MUT = 0.05

# Função para gerar indivíduos aleatoriamente
def gerar_individuos(tam_alvo):
    return np.random.randint(0, 2, tam_alvo)

# Retorna a aptidão do indivíduo
def aptidao_individuo(alvo, ind):
    return np.sum(alvo != ind)

# Seleção por roleta
def roleta(qtd_ind, apt):
    return np.random.choice(range(qtd_ind), p=apt, size=qtd_ind)

# Função de crossover
def crossover(individuos, selecao, ponto, cross):
    indice = 0
    n_ind = []
    for p in cross:
        pai_a = individuos[selecao[indice]]
        pai_b = individuos[selecao[indice+1]]
        filho_a = []
        filho_b = []
        if p <= P_CROSS:
            filho_a = np.concatenate((pai_a[:ponto], pai_b[ponto:]))
            filho_b = np.concatenate((pai_b[:ponto], pai_a[ponto:]))
        else:
            filho_a = pai_a.copy()
            filho_b = pai_b.copy()
        n_ind.append(filho_a)
        n_ind.append(filho_b)
        indice += 2
    return n_ind

# Função de mutação
def mutacao(new_individuos, tam_alvo, qtd_ind):
    for j in range(qtd_ind):
        for i in range(tam_alvo):
            p = np.random.uniform(0, 1)
            if p <= P_MUT:
                new_individuos[j][i] = 1 if new_individuos[j][i] == 0 else 0
    return new_individuos

# Função que executa um experimento completo
def executar_experimento(qtd_ind, tam_alvo, alvo, evo_acerto, geracoes_max=1000):
    individuos = [gerar_individuos(tam_alvo) for _ in range(qtd_ind)]
    
    for g in range(geracoes_max):
        apt = [aptidao_individuo(alvo, ind) for ind in individuos]
        evo_acerto.append(max(apt))
        
        # Verifica se encontrou a solução
        if max(apt) == tam_alvo:
            final_apt = [aptidao_individuo(alvo, ind) for ind in individuos]
            melhor_idx = np.argmax(final_apt)
            print(f'\nMelhor indivíduo encontrado: {individuos[melhor_idx]}')
            print(f'Alvo:                        {alvo}')
            print(f'Aptidão final: {final_apt[melhor_idx]}/{tam_alvo}')
            return g + 1  # Retorna a geração em que encontrou
        
        # Normaliza aptidão para roleta
        soma_aptidao = sum(apt)
        apt_normalizada = [x/soma_aptidao for x in apt]
        selecao = roleta(qtd_ind, apt_normalizada)
        
        # Crossover
        cross = np.random.uniform(0, 1, int(qtd_ind/2))
        ponto = np.random.randint(1, tam_alvo-1)
        individuos = crossover(individuos, selecao, ponto, cross)
        
        # Mutação
        individuos = mutacao(individuos, tam_alvo, qtd_ind)
    
    return geracoes_max  # Retorna o máximo se não encontrou

# Configurações do experimento
alvo = np.array([0,1,1,1,1,0,1,0,0,1,1,0,0,1,0])
tam_alvo = len(alvo)
qtd_ind = int(input('Selecione a quantidade de individuos a serem gerados (numero par): '))
num_experimentos = int(input('Quantos experimentos deseja executar? '))
evo_acerto = []

# Executa os experimentos
geracoes_necessarias = []
for exp in range(num_experimentos):
    geracoes = executar_experimento(qtd_ind, tam_alvo, alvo, evo_acerto)
    geracoes_necessarias.append(geracoes)

# Calcula estatísticas
media = mean(geracoes_necessarias)
desvio_padrao = stdev(geracoes_necessarias) if len(geracoes_necessarias) > 1 else 0

# Exibe resultados
print('\nResultados Finais')
print(f'Número de experimentos: {num_experimentos}')
print(f'Média de gerações necessárias: {media:.2f}')
print(f'Desvio padrão: {desvio_padrao:.2f}')
print(f'Gerações em cada rodada: {geracoes_necessarias}')

# Plota histograma das gerações necessárias
plt.hist(geracoes_necessarias, bins=20, edgecolor='black')
plt.axvline(media, color='red', linestyle='dashed', linewidth=1, label=f'Média: {media:.2f}')
plt.title("Distribuição de Gerações Necessárias")
plt.xlabel("Gerações")
plt.ylabel("Frequência")
plt.legend()
plt.show()

# Exibir gráfico de linhas + pontos
plt.scatter(range(len(evo_acerto)), evo_acerto, color='red', label='Melhor Aptidão')
plt.plot(range(len(evo_acerto)), evo_acerto, color='blue', label='Melhor Aptidão')
plt.title("Gráfico de Aptidão X Geração")
plt.xlabel("número da Geração")
plt.ylabel("aptidão")
plt.show()