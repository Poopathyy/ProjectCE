import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    return exams, rooms

exams, rooms = load_data()

exams.columns = exams.columns.str.lower()
rooms.columns = rooms.columns.str.lower()

num_timeslots = exams["exam_time"].nunique()

# =========================
# Repair Function
# =========================
def repair_solution(timeslot, room, exam_idx, schedule_map):
    students = exams.iloc[exam_idx]["num_students"]
    exam_type = exams.iloc[exam_idx]["exam_type"].lower()

    # Capacity repair
    feasible_rooms = rooms[rooms["capacity"] >= students].index.tolist()
    if room not in feasible_rooms and feasible_rooms:
        room = random.choice(feasible_rooms)

    room_type = rooms.iloc[room]["room_type"].lower()

    # Room-type repair
    if exam_type == "practical" and "lab" not in room_type:
        labs = rooms[rooms["room_type"].str.contains("lab", case=False)].index.tolist()
        if labs:
            room = random.choice(labs)

    if exam_type == "theory" and "lab" in room_type:
        non_labs = rooms[~rooms["room_type"].str.contains("lab", case=False)].index.tolist()
        if non_labs:
            room = random.choice(non_labs)

    # Room-timeslot conflict
    if (timeslot, room) in schedule_map:
        free_rooms = [
            r for r in rooms.index if (timeslot, r) not in schedule_map
        ]
        if free_rooms:
            room = random.choice(free_rooms)
        else:
            timeslot = (timeslot + 1) % num_timeslots

    return timeslot, room

# =========================
# Fitness Function (RAW)
# =========================
def fitness_ga(solution):
    penalty = 0
    room_usage = np.zeros(len(rooms))
    schedule_map = set()

    for i in range(len(exams)):
        timeslot = int(np.clip(round(solution[2*i]), 0, num_timeslots - 1))
        room = int(np.clip(round(solution[2*i + 1]), 0, len(rooms) - 1))

        students = exams.iloc[i]["num_students"]
        exam_type = exams.iloc[i]["exam_type"].lower()
        room_type = rooms.iloc[room]["room_type"].lower()
        capacity = rooms.iloc[room]["capacity"]

        if (
            students > capacity or
            (exam_type == "practical" and "lab" not in room_type) or
            (exam_type == "theory" and "lab" in room_type) or
            (timeslot, room) in schedule_map
        ):
            timeslot, room = repair_solution(timeslot, room, i, schedule_map)

        if students > capacity:
            penalty += 1
        if exam_type == "practical" and "lab" not in room_type:
            penalty += 0.5
        if exam_type == "theory" and "lab" in room_type:
            penalty += 0.5
        if (timeslot, room) in schedule_map:
            penalty += 1

        schedule_map.add((timeslot, room))
        room_usage[room] += students

    utilization_penalty = np.var(room_usage / np.sum(room_usage)) if np.sum(room_usage) > 0 else 0

    return penalty + utilization_penalty

# =========================
# GA Runner
# =========================
def run_ga(pop_size, generations, crossover_rate, mutation_rate):
    start = time.time()

    dimensions = len(exams) * 2
    population = np.random.rand(pop_size, dimensions)

    for p in range(pop_size):
        for i in range(len(exams)):
            population[p][2*i] *= num_timeslots
            population[p][2*i+1] *= len(rooms)

    fitness_vals = np.array([fitness_ga(ind) for ind in population])
    convergence = []

    for _ in range(generations):
        new_population = []

        while len(new_population) < pop_size:
            i1, i2 = random.sample(range(pop_size), 2)
            p1 = population[i1] if fitness_vals[i1] < fitness_vals[i2] else population[i2]

            i3, i4 = random.sample(range(pop_size), 2)
            p2 = population[i3] if fitness_vals[i3] < fitness_vals[i4] else population[i4]

            # Crossover
            if random.random() < crossover_rate:
                alpha = random.random()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()

            # Mutation
            for d in range(dimensions):
                if random.random() < mutation_rate:
                    if d % 2 == 0:
                        child[d] = random.uniform(0, num_timeslots)
                    else:
                        child[d] = random.uniform(0, len(rooms))

            new_population.append(child)

        population = np.array(new_population)
        fitness_vals = np.array([fitness_ga(ind) for ind in population])
        convergence.append(np.min(fitness_vals))

    best_idx = np.argmin(fitness_vals)
    runtime = time.time() - start

    return population[best_idx], fitness_vals[best_idx], convergence, runtime

# =========================
# Streamlit UI
# =========================
st.set_page_config("GA Exam Scheduling", layout="wide")
st.title("🧬 Genetic Algorithm – Exam Scheduling")

st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 20, 200, 60, 10)
generations = st.sidebar.slider("Generations", 50, 500, 150, 50)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8, 0.05)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1, 0.01)

if st.button("🚀 Run GA"):
    with st.spinner("Optimizing schedule..."):
        best_solution, raw_fitness, convergence, runtime = run_ga(
            pop_size, generations, crossover_rate, mutation_rate
        )

    st.subheader("📊 Final Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Fitness (Final Cost)", round(raw_fitness, 1))
    col2.metric("Runtime (sec)", round(runtime, 2))
    col3.metric("Generations", generations)

    st.subheader("📈 GA Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(convergence)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Convergence")
    st.pyplot(fig)
