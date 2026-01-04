import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================
df = pd.read_csv("exam_scheduling_dataset.csv")

EXAMS = df.copy()
ROOMS = df[['classroom_id', 'capacity', 'room_type']].drop_duplicates()
TIMESLOTS = df['timeslot_id'].unique().tolist()

# =========================
# GA Functions
# =========================
def create_chromosome():
    return [
        (random.choice(TIMESLOTS), random.choice(ROOMS['classroom_id'].tolist()))
        for _ in range(len(EXAMS))
    ]

def multi_objective_fitness(chromosome):
    capacity_penalty = 0
    room_type_penalty = 0
    conflict_penalty = 0
    used = {}

    for i, (slot, room) in enumerate(chromosome):
        exam = EXAMS.iloc[i]
        room_data = ROOMS[ROOMS['classroom_id'] == room].iloc[0]

        if exam['num_students'] > room_data['capacity']:
            capacity_penalty += exam['num_students'] - room_data['capacity']

        if exam['exam_type'] == "Practical" and room_data['room_type'] != "Lab":
            room_type_penalty += 1

        key = (slot, room)
        if key in used:
            conflict_penalty += 1
        else:
            used[key] = True

    return capacity_penalty, room_type_penalty, conflict_penalty

def weighted_fitness(chromosome, w1, w2, w3):
    c, r, f = multi_objective_fitness(chromosome)
    return w1*c + w2*r + w3*f

def tournament_selection(pop, k=3, w1=10, w2=20, w3=50):
    selected = random.sample(pop, k)
    return min(selected, key=lambda c: weighted_fitness(c, w1, w2, w3))

def crossover(p1, p2):
    point = random.randint(1, len(p1)-2)
    return p1[:point] + p2[point:]

def mutate(chrom, rate):
    for i in range(len(chrom)):
        if random.random() < rate:
            chrom[i] = (
                random.choice(TIMESLOTS),
                random.choice(ROOMS['classroom_id'].tolist())
            )
    return chrom

def genetic_algorithm(pop_size, generations, mutation_rate, w1, w2, w3):
    population = [create_chromosome() for _ in range(pop_size)]
    best_scores = []

    for _ in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1 = tournament_selection(population, w1=w1, w2=w2, w3=w3)
            p2 = tournament_selection(population, w1=w1, w2=w2, w3=w3)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)
        population = new_pop
        best = min(population, key=lambda c: weighted_fitness(c, w1, w2, w3))
        best_scores.append(weighted_fitness(best, w1, w2, w3))

    return best, best_scores

# =========================
# Streamlit UI
# =========================
st.title("📅 University Exam Scheduling using Genetic Algorithm")

st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Generations", 50, 300, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

st.sidebar.header("Objective Weights")
w1 = st.sidebar.slider("Capacity Penalty Weight", 1, 20, 10)
w2 = st.sidebar.slider("Room Type Penalty Weight", 1, 30, 20)
w3 = st.sidebar.slider("Conflict Penalty Weight", 10, 100, 50)

if st.button("▶ Run Genetic Algorithm"):
    with st.spinner("Optimizing exam schedule..."):
        best_solution, fitness_curve = genetic_algorithm(
            pop_size, generations, mutation_rate, w1, w2, w3
        )

    st.success("Optimization Completed!")

    # =========================
    # Convergence Plot
    # =========================
    st.subheader("📈 Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(fitness_curve)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    st.pyplot(fig)

    # =========================
    # Final Timetable
    # =========================
    st.subheader("📋 Optimized Exam Timetable")

    result = EXAMS.copy()
    result[['timeslot_id', 'classroom_id']] = best_solution

    result = result.merge(
        ROOMS, on='classroom_id', how='left', suffixes=("", "_room")
    )

    st.dataframe(result)

    # =========================
    # Objective Breakdown
    # =========================
    c, r, f = multi_objective_fitness(best_solution)
    st.subheader("🎯 Objective Breakdown")
    st.write(f"Capacity Violations: {c}")
    st.write(f"Room Type Mismatches: {r}")
    st.write(f"Room-Time Conflicts: {f}")
