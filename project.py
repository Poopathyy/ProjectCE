import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# ==================================================
# Load Datasets
# ==================================================
exams = pd.read_csv("exam_timeslot.csv")
rooms = pd.read_csv("classrooms.csv")

exams["timeslot"] = exams["exam_day"].astype(str) + "_" + exams["exam_time"].astype(str)
TIMESLOTS = exams["timeslot"].unique().tolist()

NUM_EXAMS = len(exams)
NUM_ROOMS = len(rooms)
NUM_TIMESLOTS = len(TIMESLOTS)

# ==================================================
# Genetic Algorithm Functions
# ==================================================
def create_individual():
    return [
        (random.randint(0, NUM_TIMESLOTS - 1),
         random.randint(0, NUM_ROOMS - 1))
        for _ in range(NUM_EXAMS)
    ]

def create_population(size):
    return [create_individual() for _ in range(size)]

# ---------------- Fitness Components ----------------
def fitness_components(individual):
    capacity_violation = 0
    room_conflict = 0
    room_type_penalty = 0
    unused_capacity = 0
    used = {}

    for i, (slot, room) in enumerate(individual):
        exam = exams.iloc[i]
        classroom = rooms.iloc[room]

        key = (slot, room)
        if key in used:
            room_conflict += 1
        else:
            used[key] = 1

        if exam["num_students"] > classroom["capacity"]:
            capacity_violation += exam["num_students"] - classroom["capacity"]

        if exam["exam_type"] != classroom["room_type"]:
            room_type_penalty += 1

        unused_capacity += max(0, classroom["capacity"] - exam["num_students"])

    return capacity_violation, room_conflict, room_type_penalty, unused_capacity

# ---------------- Single Objective Fitness ----------------
def fitness(individual):
    cap, conflict, type_pen, unused = fitness_components(individual)
    penalty = (1000 * cap) + (1000 * conflict) + (10 * type_pen) + unused
    return 1 / (1 + penalty)

# ---------------- Multi-Objective Fitness ----------------
def multi_objective_fitness(individual):
    cap, conflict, type_pen, unused = fitness_components(individual)
    return cap + conflict, type_pen + unused

# ==================================================
# GA Operators
# ==================================================
def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    return max(selected, key=fitness)

def crossover(p1, p2):
    point = random.randint(1, NUM_EXAMS - 1)
    return p1[:point] + p2[point:]

def mutate(individual, rate):
    for i in range(NUM_EXAMS):
        if random.random() < rate:
            individual[i] = (
                random.randint(0, NUM_TIMESLOTS - 1),
                random.randint(0, NUM_ROOMS - 1)
            )
    return individual

def genetic_algorithm(pop_size, generations, mutation_rate):
    population = create_population(pop_size)
    best_fitness_history = []

    for _ in range(generations):
        new_population = []
        for _ in range(pop_size):
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        best = max(population, key=fitness)
        best_fitness_history.append(fitness(best))

    return best, best_fitness_history, population

# ==================================================
# Pareto Front Functions
# ==================================================
def dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def pareto_front(population):
    fitnesses = [multi_objective_fitness(ind) for ind in population]
    pareto = []

    for i, f_i in enumerate(fitnesses):
        dominated = False
        for j, f_j in enumerate(fitnesses):
            if dominates(f_j, f_i):
                dominated = True
                break
        if not dominated:
            pareto.append((population[i], f_i))

    return pareto

# ==================================================
# Streamlit UI
# ==================================================
st.title("📘 University Exam Scheduling using Genetic Algorithm")

st.markdown("""
This application applies **Genetic Algorithm (GA)** to solve the **University Exam Scheduling problem**.
It evaluates feasibility and quality while visualizing **multi-objective trade-offs using a Pareto front**.
""")

# ---------------- Sidebar ----------------
st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Generations", 50, 300, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

# ---------------- Run GA ----------------
if st.button("🚀 Run Genetic Algorithm"):
    best_solution, history, final_population = genetic_algorithm(
        population_size,
        generations,
        mutation_rate
    )

    # ---------- Convergence ----------
    st.subheader("📈 Fitness Convergence")
    plt.figure()
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    st.pyplot(plt)

    # ---------- Pareto Front ----------
    st.subheader("⚖️ Pareto Front (Multi-Objective Optimization)")
    pareto = pareto_front(final_population)

    x = [p[1][0] for p in pareto]
    y = [p[1][1] for p in pareto]

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Hard Constraint Violations")
    plt.ylabel("Quality Penalty")
    plt.title("Pareto Front of Exam Scheduling Solutions")
    st.pyplot(plt)

    st.write(f"Number of Pareto-optimal solutions: **{len(pareto)}**")

    # ---------- Best Schedule ----------
    st.subheader("📋 Best Exam Schedule")
    schedule = []

    for i, (slot, room) in enumerate(best_solution):
        schedule.append({
            "Exam ID": exams.iloc[i]["exam_id"],
            "Course": exams.iloc[i]["course_code"],
            "Timeslot": TIMESLOTS[slot],
            "Room": rooms.iloc[room]["room_number"],
            "Capacity": rooms.iloc[room]["capacity"],
            "Students": exams.iloc[i]["num_students"]
        })

    st.dataframe(pd.DataFrame(schedule))

    # ---------- Constraint Summary ----------
    st.subheader("⚠️ Constraint Evaluation")
    cap, conflict, type_pen, unused = fitness_components(best_solution)

    st.write(f"Room Capacity Violations: **{cap}**")
    st.write(f"Room-Time Conflicts: **{conflict}**")
    st.write(f"Room Type Mismatches: **{type_pen}**")
    st.write(f"Unused Room Capacity: **{unused}**")
