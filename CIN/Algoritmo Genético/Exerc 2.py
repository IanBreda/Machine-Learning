import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Função Objetivo
def g(x):
    #Calcula o valor da função g(x) para um dado x.
    if not (0 <= x <= 1):
    # Retorna um valor muito baixo se x estiver fora do intervalo para penalizar soluções inválidas 
        return -np.inf
    term1 = 2**(-2 * ((x - 0.1) / 0.9)**2)
    term2 = (np.sin(5 * np.pi * x))**6
    return term1 * term2

# Parâmetros do Problema 
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
INTERVAL_RANGE = UPPER_BOUND - LOWER_BOUND

# Parâmetros do Algoritmo Genético 
# Precisão: Pelo menos 3 casas decimais (1000 pontos).
NUM_BITS = 12
MAX_INT_VALUE = 2**NUM_BITS - 1

# Parâmetros que podem ser ajustados
POPULATION_SIZE = 10     # Tamanho da população
NUM_GENERATIONS = 50    # Número de gerações
CROSSOVER_RATE = 0.6     # Taxa de crossover
MUTATION_RATE = 0.05     # Taxa de mutação (por bit)
TOURNAMENT_SIZE = 5       # Tamanho do torneio (para seleção por torneio)
ELITISM_COUNT = 1         # Número de melhores indivíduos a passar diretamente (elitismo)

def bits_to_float(bits):
    # Converte uma lista/array de bits para um float no intervalo [LOWER_BOUND, UPPER_BOUND].
    # Converte a lista de bits para um inteiro
    integer_value = 0
    for bit in bits:
        integer_value = (integer_value << 1) | bit # Equivalente a integer_value * 2 + bit

    # Mapeia o inteiro para o intervalo [0, 1] e depois para [LOWER_BOUND, UPPER_BOUND]
    float_value = LOWER_BOUND + (integer_value / MAX_INT_VALUE) * INTERVAL_RANGE
    return float_value

def create_individual():
    # Cria um indivíduo aleatório (lista de bits).
    return [random.randint(0, 1) for _ in range(NUM_BITS)]

def create_initial_population(size):
    # Cria uma população inicial de indivíduos aleatórios.
    return [create_individual() for _ in range(size)]

def calculate_fitness(individual):
    # Calcula o fitness de um indivíduo.
    x = bits_to_float(individual)
    return g(x)

def calculate_population_fitness(population):
    # Calcula o fitness de toda a população.
    return [calculate_fitness(ind) for ind in population]

# Métodos de Seleção

def selection_roulette(population, fitnesses):
    # Seleciona um pai usando o método da Roleta.
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        # Caso onde todos têm fitness 0, seleciona aleatoriamente
        return random.choice(population)

    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > pick:
            return individual
    # Fallback em caso de erro de ponto flutuante
    return population[-1]

def selection_tournament(population, fitnesses, k):
    # Seleciona um pai usando o método de Torneio.
    # Seleciona k indivíduos aleatoriamente da população
    tournament_contenders_indices = random.sample(range(len(population)), k)
    tournament_contenders = [population[i] for i in tournament_contenders_indices]
    tournament_fitnesses = [fitnesses[i] for i in tournament_contenders_indices]

    # Encontra o melhor entre os k selecionados
    winner_index = np.argmax(tournament_fitnesses)
    return tournament_contenders[winner_index]

def selection_sus(population, fitnesses, n_parents):
    # Seleciona n_parents usando Amostragem Universal Estocástica (SUS).
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        # Se todos fitness são 0, retorna n pais aleatórios
        return random.sample(population, n_parents)

    point_distance = total_fitness / n_parents
    start_point = random.uniform(0, point_distance)
    points = [start_point + i * point_distance for i in range(n_parents)]

    parents = []
    current_cumulative_fitness = 0
    pop_idx = 0
    for point in points:
        while current_cumulative_fitness < point:
            current_cumulative_fitness += fitnesses[pop_idx]
            pop_idx += 1
            if pop_idx >= len(population): # Evita IndexError se todos os pontos caírem após o último
                 pop_idx = len(population) -1
                 break # Sai do while interno se chegou ao fim
        # O pai é o indivíduo cuja faixa de fitness cumulativa contém o ponto
        # Subtrai 1 do pop_idx porque ele foi incrementado uma vez a mais no loop while
        parent_index = max(0, pop_idx - 1)
        parents.append(population[parent_index])

    # Garante que retornamos exatamente n_parents, mesmo com erros de ponto flutuante
    while len(parents) < n_parents:
         parents.append(random.choice(population)) # Adiciona aleatório se faltar

    return parents[:n_parents] # Retorna a quantidade correta


def select_parents(population, fitnesses, num_parents, method='roulette', tournament_size=3):
    # Seleciona um conjunto de pais usando o método especificado.
    parents = []
    if method == 'roulette':
        for _ in range(num_parents):
            parents.append(selection_roulette(population, fitnesses))
    elif method == 'tournament':
        for _ in range(num_parents):
            parents.append(selection_tournament(population, fitnesses, tournament_size))
    elif method == 'sus':
         # SUS seleciona todos os pais de uma vez
         parents = selection_sus(population, fitnesses, num_parents)
    else:
        raise ValueError(f"Método de seleção desconhecido: {method}")

    return parents

def crossover_one_point(parent1, parent2, rate):
    # Realiza crossover de um ponto entre dois pais.
    if random.random() < rate:
        # Escolhe um ponto de corte (sem incluir as extremidades 0 e NUM_BITS)
        crossover_point = random.randint(1, NUM_BITS - 1)
        # Cria filhos trocando as caudas
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        # Se não houver crossover, os filhos são cópias dos pais
        return parent1[:], parent2[:] # Retorna cópias

def mutate_bit_flip(individual, rate):
    # Realiza mutação por inversão de bit em um indivíduo.
    mutated_individual = individual[:] # Cria uma cópia para modificar
    for i in range(NUM_BITS):
        if random.random() < rate:
            mutated_individual[i] = 1 - mutated_individual[i] # Inverte o bit (0->1, 1->0)
    return mutated_individual

def genetic_algorithm(pop_size, num_bits, generations, crossover_rate, mutation_rate,
                      selection_method='roulette', tournament_size=3, elitism_count=1):
    # Executa o algoritmo genético.

    print(f"Iniciando AG com Seleção: {selection_method.upper()}")
    start_time = time.time()

    # 1. Inicialização
    population = create_initial_population(pop_size)
    best_overall_individual = None
    best_overall_fitness = -np.inf
    fitness_history = [] # Para guardar o melhor fitness de cada geração

    for generation in range(generations):
        # 2. Avaliação (Fitness)
        fitnesses = calculate_population_fitness(population)

        # Encontra o melhor da geração atual
        current_best_fitness = max(fitnesses)
        current_best_index = np.argmax(fitnesses)
        current_best_individual = population[current_best_index]

        # Atualiza o melhor global
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_individual = current_best_individual[:] # Guarda cópia

        fitness_history.append(best_overall_fitness)

        if (generation + 1) % 10 == 0: # Imprime progresso
             print(f"Geração {generation+1}/{generations} - Melhor Fitness: {best_overall_fitness:.6f}")

        # Prepara a próxima geração
        next_population = []

        # 3. Elitismo (preserva os melhores)
        if elitism_count > 0:
            # Ordena a população atual por fitness (do maior para o menor)
            sorted_indices = np.argsort(fitnesses)[::-1]
            elites = [population[i] for i in sorted_indices[:elitism_count]]
            next_population.extend(elites)

        # 4. Geração de novos indivíduos (Seleção, Crossover, Mutação)
        num_offspring = pop_size - elitism_count
        # Precisamos selecionar pais suficientes para gerar 'num_offspring'
        # Se geramos 2 filhos por crossover, precisamos de 'num_offspring' pais.
        # Ajuste se o crossover gerar apenas 1 filho ou se usar métodos diferentes.
        num_parents_to_select = num_offspring

        # Seleciona os pais
        parents = select_parents(population, fitnesses, num_parents_to_select,
                                 method=selection_method, tournament_size=tournament_size)

        # Cria os filhos
        offspring_count = 0
        while offspring_count < num_offspring:
             # Pega dois pais (garante que temos pares, mesmo que num_offspring seja ímpar)
             idx1 = offspring_count % len(parents)
             idx2 = (offspring_count + 1) % len(parents)
             # Garante que não são o mesmo pai se possível (pode acontecer se a seleção retornar duplicados)
             if len(parents) > 1 and idx1 == idx2:
                 idx2 = (idx2 + 1) % len(parents)

             parent1 = parents[idx1]
             parent2 = parents[idx2]

             # 5. Crossover
             child1, child2 = crossover_one_point(parent1, parent2, crossover_rate)

             # 6. Mutação
             mutated_child1 = mutate_bit_flip(child1, mutation_rate)
             mutated_child2 = mutate_bit_flip(child2, mutation_rate)

             next_population.append(mutated_child1)
             offspring_count += 1
             if offspring_count < num_offspring: # Adiciona o segundo filho se ainda couber
                 next_population.append(mutated_child2)
                 offspring_count += 1


        # 7. Nova População
        population = next_population[:pop_size] # Garante o tamanho correto

    # Fim do loop de gerações
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Decodifica o melhor indivíduo encontrado
    best_x = bits_to_float(best_overall_individual)

    print("-" * 50)
    print(f"AG ({selection_method.upper()}) Concluído!")
    print(f"Tempo de execução: {elapsed_time:.4f} segundos")
    print(f"Melhor fitness encontrado (g(x)): {best_overall_fitness:.8f}")
    print(f"Melhor x encontrado: {best_x:.8f}")
    print(f"Melhor indivíduo (bits): {''.join(map(str, best_overall_individual))}")
    print("-" * 50)

    return best_x, best_overall_fitness, fitness_history, elapsed_time

# Execução e Comparação

results = {}

# Executa com Roleta
best_x_r, best_g_r, history_r, time_r = genetic_algorithm(
    pop_size=POPULATION_SIZE, num_bits=NUM_BITS, generations=NUM_GENERATIONS,
    crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
    selection_method='roulette', elitism_count=ELITISM_COUNT
)
results['roulette'] = {'x': best_x_r, 'g(x)': best_g_r, 'time': time_r, 'history': history_r}

# Executa com Torneio
best_x_t, best_g_t, history_t, time_t = genetic_algorithm(
    pop_size=POPULATION_SIZE, num_bits=NUM_BITS, generations=NUM_GENERATIONS,
    crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
    selection_method='tournament', tournament_size=TOURNAMENT_SIZE,
    elitism_count=ELITISM_COUNT
)
results['tournament'] = {'x': best_x_t, 'g(x)': best_g_t, 'time': time_t, 'history': history_t}

# Executa com SUS
best_x_s, best_g_s, history_s, time_s = genetic_algorithm(
    pop_size=POPULATION_SIZE, num_bits=NUM_BITS, generations=NUM_GENERATIONS,
    crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
    selection_method='sus', elitism_count=ELITISM_COUNT
)
results['sus'] = {'x': best_x_s, 'g(x)': best_g_s, 'time': time_s, 'history': history_s}


# Visualização

# 1. Gráfico da Função Original e Ponto Encontrado
plt.figure(figsize=(12, 6))

# Subplot 1: Função g(x) e pontos encontrados
plt.subplot(1, 2, 1)
x_vals = np.linspace(LOWER_BOUND, UPPER_BOUND, 500)
g_vals = [g(x) for x in x_vals]
plt.plot(x_vals, g_vals, label='g(x)', color='blue')

# Marca os pontos encontrados por cada método
colors = {'roulette': 'red', 'tournament': 'green', 'sus': 'purple'}
markers = {'roulette': 'o', 'tournament': 's', 'sus': '^'}
for method, res in results.items():
    plt.scatter(res['x'], res['g(x)'], color=colors[method], marker=markers[method], s=100,
                label=f'{method.capitalize()} (x={res["x"]:.4f}, g(x)={res["g(x)"]:.4f})', zorder=5) # zorder para ficar na frente

plt.title('Função g(x) e Máximos Encontrados pelo AG')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.ylim(bottom=min(g_vals) - 0.1, top=max(g_vals) + 0.1) # Ajuste do limite inferior do eixo y
plt.legend()
plt.grid(True)

# 2. Gráfico da Convergência (Histórico do Fitness)
plt.subplot(1, 2, 2)
for method, res in results.items():
    plt.plot(range(1, NUM_GENERATIONS + 1), res['history'],
             label=f'{method.capitalize()} (final: {res["g(x)"]:.4f}, time: {res["time"]:.2f}s)',
             color=colors[method], marker=markers[method], markersize=3, linestyle='--')

plt.title('Convergência do AG (Melhor Fitness por Geração)')
plt.xlabel('Geração')
plt.ylabel('Melhor Fitness g(x)')
plt.legend()
plt.grid(True)
plt.tight_layout() # Ajusta o espaçamento entre subplots
plt.show()

# Impressão Resumo da Comparação 
print("\nResumo Comparativo")
print(f"{'Método':<15} {'Melhor g(x)':<20} {'Melhor x':<20} {'Tempo (s)':<10}")
print("-" * 65)
for method, res in results.items():
    print(f"{method.capitalize():<15} {res['g(x)']:<20.8f} {res['x']:<20.8f} {res['time']:.4f}")
print("-" * 65)