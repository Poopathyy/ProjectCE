import streamlit as st
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    return exams, rooms

exams, rooms = load_data()
exams.columns = exams.columns.str.lower()
rooms.columns = rooms.columns.str.lower()

# ==============================
# Prepare Data
# ==============================
exam_ids = exams["exam_id"].tolist()
timeslots = exams["exam_time"].unique().tolist()

num_students = dict(zip(exams["exam_id"], exams["num_students"]))
exam_type = dict(zip(exams["exam_id"], exams["exam_type"]))

room_ids = rooms["room_number"].tolist()
room_capacity = dict(zip(rooms["room_number"], rooms["capacity"]))
room_type = dict(zip(rooms["room_number"], rooms["room_type"]))

# ==============================
# Genetic Algorithm
# ==============================
def create_chromosome():
    return {e: (random.choice(timeslots), random.choice(room_ids)) for e in exam_ids}

def fitness(solution):
    penalty = 0
    schedule = {}

    for e, (ts, r) in solution.items():
        schedule.setdefault((ts, r), []).append(e)

    for (ts, r), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)

        if len(exams_here) > 1:
            penalty += 1000 * (len(exams_here) - 1)

        if students > room_capacity[r]:
            penalty += 1000

        for e in exams_here:
            if exam_type[e].lower() == "practical" and "lab" not in room_type[r].lower():
                penalty += 500
            if exam_type[e].lower() == "theory" and "lab" in room_type[r].lower():
                penalty += 300

        penalty += max(room_capacity[r] - students, 0) * 0.1

    return penalty

def selection(pop):
    return min(random.sample(pop, 3), key=fitness)

def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1.copy()
    return {e: p1[e] if random.random() < 0.5 else p2[e] for e in exam_ids}

def mutation(chromo, rate):
    for e in exam_ids:
        if random.random() < rate:
            chromo[e] = (random.choice(timeslots), random.choice(room_ids))
    return chromo

def genetic_algorithm(pop_size, gens, m_rate, c_rate):
    pop = [create_chromosome() for _ in range(pop_size)]
    history = []

    for _ in range(gens):
        pop = [
            mutation(
                crossover(selection(pop), selection(pop), c_rate),
                m_rate
            ) for _ in range(pop_size)
        ]
        history.append(fitness(min(pop, key=fitness)))

    return min(pop, key=fitness), history

# ==============================
# Metrics
# ==============================
def compute_metrics(solution, runtime):
    capacity_violations = 0
    wasted_capacity = 0
    schedule = {}

    for e, (ts, r) in solution.items():
        schedule.setdefault((ts, r), []).append(e)

    for (ts, r), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)
        if students > room_capacity[r]:
            capacity_violations += 1
        wasted_capacity += max(room_capacity[r] - students, 0)

    raw_fitness = fitness(solution)
    final_cost = round(raw_fitness / 1000, 1)

    return final_cost, capacity_violations, wasted_capacity, runtime

# ==============================
# STREAMLIT UI (NEW)
# ==============================
st.set_page_config("Exam Scheduling GA", layout="wide")

st.title("🎓 Exam Scheduling Optimization using Genetic Algorithm")
st.caption("Single-objective and Multi-objective Optimization for University Exam Timetabling")

# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙ Optimization Settings")

mode = st.sidebar.radio(
    "Optimization Mode",
    ["Single Objective", "Multi Objective"]
)

pop_size = st.sidebar.slider("Population Size", 20, 200, 50)
gens = st.sidebar.slider("Generations", 50, 500, 100)
m_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
c_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)

run = st.sidebar.button("🚀 Run Optimization")

# ==============================
# Run GA
# ==============================
if run:
    start = time.time()
    best, history = genetic_algorithm(pop_size, gens, m_rate, c_rate)
    runtime = time.time() - start

    final_cost, cap_v, wasted, runtime = compute_metrics(best, runtime)

    # ==============================
    # Metrics
    # ==============================
    st.subheader("📊 Performance Metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Cost", final_cost)
    m2.metric("Capacity Violations", cap_v)
    m3.metric("Wasted Capacity", wasted)
    m4.metric("Computation Time (s)", round(runtime, 2))

    # ==============================
    # Convergence Curve
    # ==============================
    st.subheader("📈 Convergence Curve")

    fig, ax = plt.subplots()
    ax.plot(history, linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    ax.grid(True)
    st.pyplot(fig)

    # ==============================
    # Timetable
    # ==============================
    st.subheader("🗓 Optimized Exam Timetable")

    timetable = pd.DataFrame([
        {
            "Exam ID": e,
            "Exam Type": exam_type[e],
            "Timeslot": ts,
            "Room": r,
            "Room Type": room_type[r],
            "Students": num_students[e],
            "Capacity": room_capacity[r]
        }
        for e, (ts, r) in best.items()
    ])

    st.dataframe(timetable, use_container_width=True)

st.markdown("---")
st.markdown("**Course:** JIE42903 – Evolutionary Computing  \n**Method:** Genetic Algorithm")
