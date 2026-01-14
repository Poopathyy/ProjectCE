import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import time

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    return exams, rooms

exams, rooms = load_data()

# Normalize columns
exams.columns = exams.columns.str.lower()
rooms.columns = rooms.columns.str.lower()

# ==============================
# Prepare Data
# ==============================
exam_ids = exams["exam_id"].tolist()
timeslots = exams["exam_time"].unique().tolist()
room_ids = rooms["room_number"].tolist()

num_students = dict(zip(exams["exam_id"], exams["num_students"]))
exam_type = dict(zip(exams["exam_id"], exams["exam_type"].str.lower()))
room_capacity = dict(zip(rooms["room_number"], rooms["capacity"]))
room_type = dict(zip(rooms["room_number"], rooms["room_type"].str.lower()))
building_map = dict(zip(rooms["room_number"], rooms["building_name"]))

# ==============================
# GA Functions
# ==============================
def create_chromosome():
    return {
        exam: (random.choice(timeslots), random.choice(room_ids))
        for exam in exam_ids
    }

def fitness(chromosome):
    penalty = 0
    schedule = {}

    for exam, (ts, room) in chromosome.items():
        schedule.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in schedule.items():
        students = sum(num_students[e] for e in exams_in_room)

        # Room-timeslot conflict
        if len(exams_in_room) > 1:
            penalty += 5 * (len(exams_in_room) - 1)

        # Capacity constraint
        if students > room_capacity[room]:
            penalty += 5

        # Room type compatibility
        for e in exams_in_room:
            if exam_type[e] == "practical" and "lab" not in room_type[room]:
                penalty += 3
            if exam_type[e] == "theory" and "lab" in room_type[room]:
                penalty += 2

        # Wasted capacity (soft)
        penalty += max(room_capacity[room] - students, 0) * 0.01

    return penalty

def fitness_multi(chromosome, w1, w2, w3):
    penalty = 0
    schedule = {}

    for exam, (ts, room) in chromosome.items():
        schedule.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in schedule.items():
        students = sum(num_students[e] for e in exams_in_room)

        if len(exams_in_room) > 1:
            penalty += w1 * (len(exams_in_room) - 1)

        if students > room_capacity[room]:
            penalty += w2

        penalty += w3 * max(room_capacity[room] - students, 0)

    return penalty

def selection(population):
    candidates = random.sample(population, 3)
    candidates.sort(key=fitness)
    return candidates[0]

def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1.copy()
    return {
        exam: p1[exam] if random.random() < 0.5 else p2[exam]
        for exam in exam_ids
    }

def mutation(chromosome, rate):
    for exam in exam_ids:
        if random.random() < rate:
            chromosome[exam] = (
                random.choice(timeslots),
                random.choice(room_ids)
            )
    return chromosome

def evaluate_metrics(chromosome):
    capacity_violations = 0
    wasted_capacity = 0
    schedule = {}

    for exam, (ts, room) in chromosome.items():
        schedule.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in schedule.items():
        students = sum(num_students[e] for e in exams_in_room)
        if students > room_capacity[room]:
            capacity_violations += 1
        wasted_capacity += max(room_capacity[room] - students, 0)

    return capacity_violations, wasted_capacity

def genetic_algorithm(pop, gen, m_rate, c_rate, mode, w1, w2, w3):
    population = [create_chromosome() for _ in range(pop)]
    history = []

    for _ in range(gen):
        new_pop = []
        for _ in range(pop):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2, c_rate)
            child = mutation(child, m_rate)
            new_pop.append(child)
        population = new_pop

        best = min(
            population,
            key=lambda x: fitness_multi(x, w1, w2, w3)
            if mode == "Multi Objective" else fitness(x)
        )
        score = (
            fitness_multi(best, w1, w2, w3)
            if mode == "Multi Objective" else fitness(best)
        )
        history.append(score)

    return best, history

# ==============================
# Streamlit UI
# ==============================
st.set_page_config("Exam Scheduling GA", layout="wide")
st.title("🎓 University Exam Scheduling using Genetic Algorithm")

# Sidebar
st.sidebar.header("GA Parameters")
population = st.sidebar.slider("Population Size", 20, 200, 50, 10)
generations = st.sidebar.slider("Generations", 50, 500, 100, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1, 0.01)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8, 0.05)

mode = st.sidebar.radio("Optimization Mode", ["Single Objective", "Multi Objective"])

if mode == "Multi Objective":
    w1 = st.sidebar.slider("Clash Weight", 1, 10, 5)
    w2 = st.sidebar.slider("Capacity Weight", 1, 10, 5)
    w3 = st.sidebar.slider("Wastage Weight", 0.01, 1.0, 0.05)
else:
    w1 = w2 = w3 = None

# Run GA
if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing..."):
        start = time.time()
        best, history = genetic_algorithm(
            population, generations,
            mutation_rate, crossover_rate,
            mode, w1, w2, w3
        )
        runtime = time.time() - start

    raw_fitness = (
        fitness_multi(best, w1, w2, w3)
        if mode == "Multi Objective" else fitness(best)
    )

    capacity_violations, wasted_capacity = evaluate_metrics(best)

    # ==============================
    # Metrics Display
    # ==============================
    st.subheader("📊 Final Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Final Cost", round(raw_fitness, 1))
    c2.metric("Capacity Violations", capacity_violations)
    c3.metric("Wasted Capacity", wasted_capacity)
    c4.metric("Computation Time (s)", round(runtime, 2))

    # ==============================
    # Convergence Curve
    # ==============================
    st.subheader("📈 Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    # ==============================
    # Timetable
    # ==============================
    st.subheader("🗓 Optimized Exam Timetable")
    timetable = pd.DataFrame([
        {
            "Exam ID": e,
            "Time Slot": ts,
            "Room": r,
            "Building": building_map[r],
            "Students": num_students[e],
            "Capacity": room_capacity[r]
        }
        for e, (ts, r) in best.items()
    ])
    st.dataframe(timetable, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Course:** JIE42903 – Evolutionary Computing  \n**Method:** Genetic Algorithm")
