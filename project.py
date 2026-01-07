import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# =========================
# Load datasets
# =========================
@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    return exams, rooms

exams, rooms = load_data()
ROOMS = rooms["room_id"].tolist()
TIMESLOTS = list(range(1, 11))

# =========================
# Streamlit UI
# =========================
st.title("🎓 Multi-Objective GA for University Exam Scheduling")

# -------------------------
# GA Parameters
# -------------------------
st.sidebar.header("GA Parameters")
POP_SIZE = st.sidebar.slider("Population Size", 50, 300, 120)
GENERATIONS = st.sidebar.slider("Generations", 50, 400, 200)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.8)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.1)

# -------------------------
# Objective Weights
# -------------------------
st.sidebar.header("🎯 Objective Weights")

W_HARD = st.sidebar.slider(
    "Hard Constraint Weight (Highest Priority)",
    1.0, 10.0, 5.0, step=0.5
)

W_SOFT = st.sidebar.slider(
    "Soft Constraint Weight",
    0.1, 5.0, 1.0, step=0.1
)

W_UTIL = st.sidebar.slider(
    "Room Utilization Weight",
    0.1, 5.0, 1.0, step=0.1
)

# =========================
# Dataset Overview
# =========================
st.header("📊 Dataset Overview")
st.subheader("Exams")
st.dataframe(exams)
st.subheader("Classrooms")
st.dataframe(rooms)

# =========================
# GA Representation
# =========================
def create_chromosome():
    return [(random.choice(TIMESLOTS), random.choice(ROOMS))
            for _ in range(len(exams))]

def create_population():
    return [create_chromosome() for _ in range(POP_SIZE)]

# =========================
# Multi-objective Evaluation
# =========================
def evaluate(chromosome):
    hard_penalty = 0
    soft_penalty = 0
    utilization = 0

    schedule = {}

    for i, (ts, room) in enumerate(chromosome):
        students = exams.iloc[i]["students"]
        schedule.setdefault((ts, room), []).append(students)

    for (ts, room), student_list in schedule.items():
        capacity = rooms.loc[
            rooms["room_id"] == room, "capacity"
        ].values[0]

        # Hard constraints
        if len(student_list) > 1:
            hard_penalty += 1000

        for s in student_list:
            if s > capacity:
                hard_penalty += 1000
            utilization += s / capacity

    # Soft constraint: overloaded timeslots
    timeslot_count = {}
    for ts, _ in chromosome:
        timeslot_count[ts] = timeslot_count.get(ts, 0) + 1

    for count in timeslot_count.values():
        if count > 3:
            soft_penalty += (count - 3) * 10

    return hard_penalty, soft_penalty, -utilization

# =========================
# Pareto dominance
# =========================
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_nondominated_sort(population, scores):
    fronts = [[]]
    domination_count = [0] * len(population)
    dominated = [[] for _ in range(len(population))]

    for i in range(len(population)):
        for j in range(len(population)):
            if dominates(scores[i], scores[j]):
                dominated[i].append(j)
            elif dominates(scores[j], scores[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]

# =========================
# Selection
# =========================
def selection(population, scores):
    fronts = fast_nondominated_sort(population, scores)
    selected = []
    for front in fronts:
        for idx in front:
            selected.append(population[idx])
            if len(selected) == POP_SIZE:
                return selected
    return selected

# =========================
# Crossover & Mutation
# =========================
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1, p2

def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = (random.choice(TIMESLOTS), random.choice(ROOMS))
    return chromosome

# =========================
# Run GA
# =========================
if st.button("🚀 Run Multi-Objective GA"):

    start = time.time()
    population = create_population()

    for _ in range(GENERATIONS):
        scores = [evaluate(c) for c in population]
        selected = selection(population, scores)

        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            new_pop.append(mutate(c2))

        population = new_pop[:POP_SIZE]

    scores = [evaluate(c) for c in population]
    fronts = fast_nondominated_sort(population, scores)
    pareto_front = fronts[0]

    # =========================
    # Pareto Front Visualization
    # =========================
    st.header("📈 Pareto Front")

    pareto_scores = [scores[i] for i in pareto_front]
    x = [s[1] for s in pareto_scores]      # soft penalty
    y = [-s[2] for s in pareto_scores]     # utilization

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Soft Constraint Penalty")
    ax.set_ylabel("Room Utilization")
    st.pyplot(fig)

    # =========================
    # Weighted Pareto Selection
    # =========================
    def weighted_score(s):
        return (
            W_HARD * s[0] +
            W_SOFT * s[1] +
            W_UTIL * s[2]
        )

    best_idx = min(pareto_front, key=lambda i: weighted_score(scores[i]))
    best_solution = population[best_idx]

    runtime = time.time() - start

    st.metric("Pareto Solutions", len(pareto_front))
    st.metric("Execution Time (seconds)", round(runtime, 2))

    # =========================
    # Final Timetable
    # =========================
    st.header("🗓️ Selected Timetable (Based on Weights)")

    timetable = []
    for i, (ts, room) in enumerate(best_solution):
        timetable.append({
            "Exam ID": exams.iloc[i]["exam_id"],
            "Students": exams.iloc[i]["students"],
            "Timeslot": ts,
            "Room": room
        })

    df = pd.DataFrame(timetable)
    st.dataframe(df)

    st.download_button(
        "📥 Download Timetable",
        df.to_csv(index=False),
        "weighted_pareto_exam_timetable.csv",
        "text/csv"
    )

    st.markdown("---")
    st.markdown("**Multi-Objective GA with Interactive Weights | JIE42903**")
