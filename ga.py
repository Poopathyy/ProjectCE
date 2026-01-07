import random
import time
from fitness import fitness


def create_individual(num_exams, num_rooms):
    return [random.randint(0, num_rooms - 1) for _ in range(num_exams)]


def create_population(size, num_exams, num_rooms):
    return [create_individual(num_exams, num_rooms) for _ in range(size)]


def tournament_selection(population, exams, rooms, k=3):
    selected = random.sample(population, k)
    selected.sort(
        key=lambda c: fitness(c, exams, rooms),
        reverse=True
    )
    return selected[0]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    return parent1[:point] + parent2[point:]


def mutate(chromosome, num_rooms, rate):
    for i in range(len(chromosome)):
        if random.random() < rate:
            chromosome[i] = random.randint(0, num_rooms - 1)
    return chromosome


def genetic_algorithm(
    exams,
    rooms,
    population_size=50,
    generations=100,
    mutation_rate=0.1
):
    start_time = time.time()

    num_exams = len(exams)
    num_rooms = len(rooms)

    population = create_population(population_size, num_exams, num_rooms)

    best_history = []
    avg_history = []

    for _ in range(generations):
        new_population = []

        fitness_vals = [fitness(c, exams, rooms) for c in population]
        avg_history.append(sum(fitness_vals) / len(fitness_vals))

        for _ in range(population_size):
            p1 = tournament_selection(population, exams, rooms)
            p2 = tournament_selection(population, exams, rooms)

            child = crossover(p1, p2)
            child = mutate(child, num_rooms, mutation_rate)
            new_population.append(child)

        population = new_population
        best = max(population, key=lambda c: fitness(c, exams, rooms))
        best_history.append(fitness(best, exams, rooms))

    runtime = time.time() - start_time
    return best, best_history, avg_history, runtime
