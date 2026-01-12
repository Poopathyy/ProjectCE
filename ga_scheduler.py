import random
from utils import random_assignment, generate_timeslots
from fitness import fitness_function

def selection(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

def crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:]

def mutation(chromosome, timeslots, rooms, rate=0.1):
    for gene in chromosome:
        if random.random() < rate:
            gene["timeslot"] = random.choice(timeslots)
            gene["room"] = random.choice(rooms)
    return chromosome

def run_ga(exams, rooms, generations=200, pop_size=50):
    timeslots = generate_timeslots()
    population = [
        random_assignment(exams, rooms, timeslots)
        for _ in range(pop_size)
    ]

    best_solution = None
    best_fitness = 0
    history = []

    for _ in range(generations):
        fitnesses = [fitness_function(ind, rooms) for ind in population]
        history.append(max(fitnesses))

        new_population = []
        for _ in range(pop_size):
            p1, p2 = selection(population, fitnesses)
            child = crossover(p1, p2)
            child = mutation(child, timeslots, rooms["room_id"].tolist())
            new_population.append(child)

        population = new_population

        max_fit = max(fitnesses)
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_solution = population[fitnesses.index(max_fit)]

    return best_solution, history
