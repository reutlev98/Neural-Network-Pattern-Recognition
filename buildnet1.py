import copy
import random
import json
import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE = 16
HIDDEN1 = 2
OUTPUT_SIZE = 1
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
MUTATION_RATE = 0.15
BEST = 10


def read_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.read().splitlines()
    features = []
    labels = []
    for line in lines:  # Split the lines into features and labels
        binary_string, label = line.split()
        features.append(list(map(int, binary_string)))
        labels.append(int(label))
    data = list(zip(features, labels))  # Combine features and labels into a list of tuples
    return data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xavier(shape):
    limit = np.sqrt(6 / (shape[1] + shape[0]))
    return np.random.uniform(-limit, limit, shape)


class Network:
    def __init__(self, structure, weights=None):
        self.structure = structure
        if weights is None:
            self.weights = [xavier((structure[i + 1], structure[i])) for i in range(len(structure) - 1)]
        else:
            self.weights = weights

    def predict(self, item):
        data_temp = item
        for layer_weights in self.weights[:]:
            data_temp = sigmoid((np.dot(layer_weights, data_temp)))
        return data_temp


def population_init(population_size, input_size, output_size):
    population = []
    for _ in range(population_size):
        struct = [input_size, HIDDEN1, output_size]  # Fixed network structure
        population.append(Network(struct))
    return population


def cal_fitness(network, data):
    correct = 0
    for item in data:
        input_data = np.array(item[0])
        true_label = item[1]
        output = network.predict(input_data)
        pred_label = 1 if output > 0.5 else 0
        if pred_label == true_label:
            correct += 1
    fitness = correct / len(data)
    return fitness


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def crossover(parent1, parent2):
    weights1 = []
    weights2 = []
    for i in range(len(parent1.weights)):
        rows, cols = parent1.weights[i].shape
        offspring1 = np.zeros((rows, cols))  # Perform uniform crossover
        offspring2 = np.zeros((rows, cols))
        for row in range(rows):
            for col in range(cols):
                if random.random() < 0.5:
                    offspring1[row, col] = parent1.weights[i][row, col]
                    offspring2[row, col] = parent2.weights[i][row, col]
                else:
                    offspring1[row, col] = parent2.weights[i][row, col]
                    offspring2[row, col] = parent1.weights[i][row, col]
        weights1.append(offspring1)
        weights2.append(offspring2)
    children = [Network(parent1.structure, weights1), Network(parent1.structure, weights2)]
    return children


def mutate(network):
    mutated_weights = []
    for original_weight_matrix in network.weights:
        mutated_matrix = original_weight_matrix.copy()
        rows, cols = mutated_matrix.shape
        for row in range(rows):
            for col in range(cols):
                if random.random() <= MUTATION_RATE:
                    delta = np.random.uniform(-0.1, 0.1)  # Use a uniform distribution for delta calculation between -0.1 and 0.1
                    mutated_matrix[row, col] += delta
                    mutated_matrix[row, col] = np.clip(mutated_matrix[row, col], -1, 1) # Clip the mutated value to the range of -1 to 1
        mutated_weights.append(mutated_matrix)
    mutated_network = Network(network.structure, mutated_weights)
    return mutated_network


def evolve(population, train_data):
    global best_network
    fitness_array = [(network, cal_fitness(network, train_data)) for network in population]
    fitness_array.sort(key=lambda x: x[1], reverse=True)
    best_individuals = fitness_array[:BEST]
    population = [network for network, f in best_individuals]
    fitness = [f for network, f in best_individuals]
    best_network = best_individuals[0][0]
    print("Best fitness:", best_individuals[0][1])
    new_population = []

    # Extract the best 10% weights and add them to the new_population
    for network, _ in best_individuals:
        new_population.append(copy.deepcopy(network))

    # Create one copy for each weight in the best 10% and perform mutations
    for network, _ in best_individuals:
        for _ in range(1):  #to delete!!!!!!!!!!!!!!!!
            mutated_network = mutate(network)
            new_population.append(mutated_network)

    # Perform crossovers to create 70 new weights out of the best 10
    for _ in range(35):
        parent_indices = np.random.choice(range(len(population)), size=2, replace=True, p=fitness / np.sum(fitness))
        parents = [population[idx] for idx in parent_indices]
        parent1, parent2 = parents[0], parents[1]
        offspring_one, offspring_two = crossover(parent1, parent2)
        offspring_one = mutate(offspring_one)
        offspring_two = mutate(offspring_two)
        new_population.append(offspring_one)
        new_population.append(offspring_two)

    # Create 10 new random weights
    for _ in range(10):
        structure = [INPUT_SIZE, HIDDEN1, OUTPUT_SIZE]  # Fixed network structure
        random_network = Network(structure)
        new_population.append(random_network)

    return new_population


train_data = read_file("train1.txt")  # Split the data into training and test sets
test_data = read_file("test1.txt")

population = population_init(POPULATION_SIZE, INPUT_SIZE, OUTPUT_SIZE)
current_best_fitness = 0
best_fitness = 0
unchanged_gens = 0
train_fitness_scores = []
test_fitness_scores = []

for generation in range(NUM_GENERATIONS):  # main loop
    print("Generation number:", generation + 1, "unchanged_gens: ", unchanged_gens)
    population = evolve(population, train_data)

    if generation % 10 == 0:  # lamarck optimization once in  10 generations
        optimized_population = []  # make lamarck optimization
        optimized_fitness = []
        for individual in population:
            temp = mutate(copy.deepcopy(individual))
            mutated = mutate(temp)
            fitness_original = cal_fitness(individual, train_data)
            fitness_mutated = cal_fitness(mutated, train_data)
            if fitness_mutated > fitness_original:
                optimized_population.append(mutated)
                optimized_fitness.append(fitness_mutated)
            else:
                optimized_population.append(individual)
                optimized_fitness.append(fitness_original)
        best_fitness = current_best_fitness

        # choose the best individual
        highest_fitness_index = np.argmax(optimized_fitness)
        current_best_fitness = optimized_fitness[highest_fitness_index]
        best_network = optimized_population[highest_fitness_index]
    else:
        best_fitness = current_best_fitness
        current_best_fitness = cal_fitness(best_network, train_data)

    if current_best_fitness > best_fitness:  # if the max fitness increased reset the counter and the mutation rate
        unchanged_gens = 0
        MUTATION_RATE = 0.15
    else:
        unchanged_gens += 1
        MUTATION_RATE = 0.15
        if unchanged_gens % 4 == 0:
            MUTATION_RATE = 0.25
    train_fitness_scores.append(current_best_fitness)
    test_accuracy = cal_fitness(best_network, test_data)  # Evaluate the best network on the test data
    test_fitness_scores.append(test_accuracy)

test_accuracy = cal_fitness(best_network, test_data)  # Evaluate the best network on the test data

print("Test Accuracy:", test_accuracy)

best_solution_with_variables = { # Save best solution and additional variables to wnet0.json
    'best_solution': best_network.weights,
    'structure': best_network.structure
}
population_json = json.dumps(best_solution_with_variables, default=convert_ndarray_to_list)

with open("wnet1.json", "w") as json_file:
    json_file.write(population_json)

# Generate the fitness score graph
generations = range(1, NUM_GENERATIONS + 1)

plt.plot(generations, train_fitness_scores, label='Training Data')
plt.plot(generations, test_fitness_scores, label='Test Data')
plt.xlabel('Generation')
plt.ylabel('Fitness Score')
plt.title('Fitness Score over Generations')
plt.legend()
plt.show()