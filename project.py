import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# =========================
# Load datasets
# =========================
exams = pd.read_csv("exam_timeslot.csv")
rooms = pd.read_csv("classrooms.csv")

exams["timeslot"] = exams["exam_day"].astype(str) + "_" + exams["exam_time"].astype(str)
TIMESLOTS = exams["timeslot"].unique().tolist()

NUM_EXAMS = len(exams)
NUM_ROOMS = len(rooms)
NUM_TIMESLOTS = len(TIMESLOTS)

# =========================
# GA Helper Functions
# =========================
def create_individual():
    return [
        (random.randint(0, NUM_TIMESLOTS - 1),
         random.randint(0, NUM_ROOMS - 1))
        for _ in range(NUM_EXAMS)
    ]

def create_population(size):
    return [create_individual() for _ in range(size)]

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

def fitness(individual):
    cap, conflict, type_pen, unused = fitness_components(individual)
    penalty = (1000 * cap) + (1000 * conflict) + (10 * type_pen) + unused
    return 1 / (1 + penalty)

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

    return best, best_fitness_history

# =========================
# Streamlit UI
# =========================
st.title("📘 University Exam Scheduling using Genetic Algorithm")

st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Number of Generations", 50, 300, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

if st.button("🚀 Run Genetic Algorithm"):
    best_solution, history = genetic_algorithm(
        population_size,
        generations,
        mutation_rate
    )

    st.subheader("📈 Fitness Convergence Curve")
    plt.figure()
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    st.pyplot(plt)

    st.subheader("📋 Optimized Exam Schedule")

    schedule = []
    for i, (slot, room) in enumerate(best_solution):
        schedule.append({
            "Exam ID": exams.iloc[i]["exam_id"],
            "Course": exams.iloc[i]["course_code"],
            "Timeslot": TIMESLOTS[slot],
            "Room": rooms.iloc[room]["room_number"],
            "Room Capacity": rooms.iloc[room]["capacity"],
            "Students": exams.iloc[i]["num_students"]
        })

    st.dataframe(pd.DataFrame(schedule))

    st.subheader("⚠️ Constraint Summary")
    cap, conflict, type_pen, unused = fitness_components(best_solution)

    st.write(f"Room Capacity Violations: **{cap}**")
    st.write(f"Room-Time Conflicts: **{conflict}**")
    st.write(f"Room Type Mismatches: **{type_pen}**")
    st.write(f"Unused Room Capacity: **{unused}**")
