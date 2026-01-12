import random
from fitness import fitness_function
from utils import random_assignment

def selection(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    return parent1[:point] + parent2[point:]

def mutation(chromosome, timeslots, rooms, rate=0.1):
    for i in range(len(chromosome)):
        if random.random() < rate:
            exam, _, _ = chromosome[i]
            chromosome[i] = (
                exam,
                random.choice(timeslots),
                random.choice(rooms)
            )
    return chromosome

def run_ga(exams, rooms, timeslots, generations=200, pop_size=50):
    population = [
        random_assignment(exams, rooms, timeslots)
        for _ in range(pop_size)
    ]

    best_solution = None
    best_fitness = 0
    history = []

    for gen in range(generations):
        fitnesses = [
            fitness_function(ind, exams, rooms)
            for ind in population
        ]

        history.append(max(fitnesses))

        new_population = []
        for _ in range(pop_size):
            p1, p2 = selection(population, fitnesses)
            child = crossover(p1, p2)
            child = mutation(
                child,
                timeslots,
                rooms['room_id'].tolist()
            )
            new_population.append(child)

        population = new_population

        if max(fitnesses) > best_fitness:
            best_fitness = max(fitnesses)
            best_solution = population[fitnesses.index(best_fitness)]

    return best_solution, history
