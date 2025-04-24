import numpy as np
import matplotlib.pyplot as plt
import time

# Definição da Função Objetivo (Rosenbrock)
def rosenbrock(params):
    # Calcula o valor da função de Rosenbrock para [x, y].
    x, y = params
    return (1 - x)**2 + 100 * (y - x**2)**2

# Intervalo de busca para x e y
LIMITES = np.array([[-10.0, 10.0], [-10.0, 10.0]]) # numpy precisa de floats
DIMENSOES = 2

# Função auxiliar para aplicar limites após operações
def aplicar_limites(individuo, limites):
    # Garante que os valores fiquem dentro dos limites 
    return np.clip(individuo, limites[:, 0], limites[:, 1])

# Inicialização da População
def inicializar_populacao(tamanho_populacao, limites):
    # Cria uma população inicial aleatória 
    populacao = np.random.uniform(limites[:, 0], limites[:, 1], size=(tamanho_populacao, DIMENSOES))
    return populacao

# Avaliação da População (Fitness é o valor da função)
def avaliar_populacao(populacao):
    # Calcula o fitness (valor da função Rosenbrock) para cada indivíduo
    # Usamos list comprehension para aplicar a função a cada linha (indivíduo)
    return np.array([rosenbrock(ind) for ind in populacao])

# Seleção por Torneio
def selecionar_pai(populacao, fitnesses, tamanho_torneio):
    # Seleciona um pai usando torneio (minimiza fitness) 
    # Escolhe K indivíduos aleatoriamente
    indices_torneio = np.random.choice(len(populacao), tamanho_torneio, replace=False)
    fitnesses_torneio = fitnesses[indices_torneio]

    # Encontra o índice do competidor com MENOR fitness dentro do torneio
    indice_vencedor_no_torneio = np.argmin(fitnesses_torneio)
    indice_vencedor_original = indices_torneio[indice_vencedor_no_torneio]

    return populacao[indice_vencedor_original]

# Cruzamento 
def cruzar(pai1, pai2, limites):
    # Realiza cruzamento de ponto único entre dois pais.
    # Ponto de corte 
    ponto_corte = np.random.randint(0, DIMENSOES)

    filho1 = np.copy(pai1)
    filho2 = np.copy(pai2)

    # Troca os segmentos após o ponto de corte
    filho1[ponto_corte:], filho2[ponto_corte:] = filho2[ponto_corte:], filho1[ponto_corte:].copy() 

    filho1 = aplicar_limites(filho1, limites)
    filho2 = aplicar_limites(filho2, limites)

    return filho1, filho2

# Mutação (Adição de Ruído Gaussiano)
def mutar(individuo, limites, taxa_mutacao, intensidade_mutacao):
    # Aplica mutação a um indivíduo com certa probabilidade."""
    if np.random.rand() < taxa_mutacao:
        # Adiciona ruído Gaussiano proporcional ao intervalo e intensidade
        faixas = limites[:, 1] - limites[:, 0]
        ruido = np.random.normal(0, intensidade_mutacao * faixas)
        individuo_mutado = individuo + ruido
        individuo_mutado = aplicar_limites(individuo_mutado, limites)
        return individuo_mutado
    return individuo # Retorna o indivíduo original se não mutar

# Função Principal do Algoritmo Genético

def executar(tamanho_populacao, num_geracoes, limites, taxa_mutacao, intensidade_mutacao, tamanho_torneio, num_elitismo):
    # Executa o AG para minimizar Rosenbrock

    populacao = inicializar_populacao(tamanho_populacao, limites)
    melhor_fitness_historia = []
    melhor_individuo_geral = None
    melhor_fitness_geral = np.inf # Inicializa com valor alto para minimização

    for geracao in range(num_geracoes):
        fitnesses = avaliar_populacao(populacao)

        # Encontra e registra o melhor da geração
        melhor_idx_geracao = np.argmin(fitnesses)
        melhor_fitness_geracao = fitnesses[melhor_idx_geracao]
        melhor_individuo_geracao = populacao[melhor_idx_geracao].copy()

        # Atualiza o melhor global
        if melhor_fitness_geracao < melhor_fitness_geral:
            melhor_fitness_geral = melhor_fitness_geracao
            melhor_individuo_geral = melhor_individuo_geracao.copy()

        melhor_fitness_historia.append(melhor_fitness_geracao)

        nova_populacao = []

        # Elitismo: Adiciona os melhores indivíduos diretamente à próxima geração
        if num_elitismo > 0:
            indices_elite = np.argsort(fitnesses)[:num_elitismo]
            elite = populacao[indices_elite].copy()
            nova_populacao.extend(elite)

        # Criação dos novos indivíduos (seleção, cruzamento, mutação)
        while len(nova_populacao) < tamanho_populacao:
            # Seleciona dois pais
            pai1 = selecionar_pai(populacao, fitnesses, tamanho_torneio)
            pai2 = selecionar_pai(populacao, fitnesses, tamanho_torneio) # Pode selecionar o mesmo pai

            # Realiza cruzamento para gerar 2 filhos
            filho1, filho2 = cruzar(pai1, pai2, limites)

            # Aplica mutação aos filhos
            filho1 = mutar(filho1, limites, taxa_mutacao, intensidade_mutacao)
            filho2 = mutar(filho2, limites, taxa_mutacao, intensidade_mutacao)

            # Adiciona filhos à nova população, cuidando para não ultrapassar o tamanho
            nova_populacao.append(filho1)
            if len(nova_populacao) < tamanho_populacao:
                nova_populacao.append(filho2)

        # Substitui a população antiga pela nova
        populacao = np.array(nova_populacao)

    return melhor_fitness_historia, melhor_individuo_geral, melhor_fitness_geral

# Configuração e Execução

TAMANHO_POPULACAO = 150
NUM_GERACOES = 1000
TAXA_MUTACAO = 0.3 # Aumentando um pouco a chance de mutação
INTENSIDADE_MUTACAO = 0.1 # Escala do ruído de mutação (0.1 * range)
TAMANHO_TORNEIO = 20
NUM_ELITISMO = 0 # Usando elitismo

print("Iniciando Algoritmo Genético...\n")


iteracoes = 1
conjunto_melhores_individuos = []
conjunto_valores_minimizados = []
melhor_historias_fit = []
fit_menor = np.inf

for i in range(iteracoes):
    start = time.time()
    historia_fitness, melhor_individuo, melhor_fitness = executar(
        TAMANHO_POPULACAO,
        NUM_GERACOES,
        LIMITES,
        TAXA_MUTACAO,
        INTENSIDADE_MUTACAO,
        TAMANHO_TORNEIO,
        NUM_ELITISMO
    )
    end = time.time()
    tempo_decorrido = end - start
    conjunto_melhores_individuos.append(melhor_individuo)
    conjunto_valores_minimizados.append(melhor_fitness)
    if melhor_fitness < fit_menor:
        melhor_historias_fit = historia_fitness.copy()
        fit_menor = melhor_fitness

    print(f"Execução {i+1} Concluída.")
    print(f"Melhor fitness encontrado: {melhor_fitness:.2e}") # Usando notação científica para valores pequenos
    print(f"Melhor indivíduo [x, y]: {melhor_individuo}")
    print(f"Valor da função Rosenbrock neste ponto: {rosenbrock(melhor_individuo):.2e}") # Verifica se o fitness corresponde
    print(f"Tempo necessário de execução: {tempo_decorrido:.2f} segundos\n")
    print('-' * 50)

best_index = np.argmin(conjunto_valores_minimizados)
print(f'Melhor individuo geral: {conjunto_melhores_individuos[best_index]}')
print(f'Valor de minimização de rosenbrock: {conjunto_valores_minimizados[best_index]:2e}')

# Visualização
plt.figure(figsize=(10, 6))
plt.plot(melhor_historias_fit)
plt.title("Convergência do AG na Minimização de Rosenbrock")
plt.xlabel("Geração")
plt.ylabel("Melhor Fitness (Valor da Função)")
plt.yscale('log') # Escala logarítmica em base 10 para melhor visualização
plt.grid(True)
plt.show()