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
# Chromosome
# ==============================
def create_chromosome():
    return {e: (random.choice(timeslots), random.choice(room_ids)) for e in exam_ids}

# ==============================
# SINGLE-OBJECTIVE FITNESS
# ==============================
def fitness(solution):
    penalty = 0
    schedule = {}

    for e, (ts, r) in solution.items():
        schedule.setdefault((ts, r), []).append(e)

    for (ts, r), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)

        # Hard constraints
        if len(exams_here) > 1:
            penalty += 1000 * (len(exams_here) - 1)

        if students > room_capacity[r]:
            penalty += 1000

        for e in exams_here:
            if exam_type[e].lower() == "practical" and "lab" not in room_type[r].lower():
                penalty += 500
            if exam_type[e].lower() == "theory" and "lab" in room_type[r].lower():
                penalty += 300

        # Soft constraint
        penalty += max(room_capacity[r] - students, 0) * 0.1

    return penalty

# ==============================
# MULTI-OBJECTIVE FITNESS
# ==============================
def multi_objective_fitness(solution):
    capacity_v = 0
    type_v = 0
    waste = 0
    schedule = {}

    for e, (ts, r) in solution.items():
        schedule.setdefault((ts, r), []).append(e)

    for (ts, r), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)

        if students > room_capacity[r]:
            capacity_v += 1

        for e in exams_here:
            if exam_type[e].lower() == "practical" and "lab" not in room_type[r].lower():
                type_v += 1
            if exam_type[e].lower() == "theory" and "lab" in room_type[r].lower():
                type_v += 1

        waste += max(room_capacity[r] - students, 0)

    return capacity_v, type_v, waste

# ==============================
# GA OPERATORS
# ==============================
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

# ==============================
# SINGLE-OBJECTIVE GA
# ==============================
def genetic_algorithm(pop_size, gens, m_rate, c_rate):
    pop = [create_chromosome() for _ in range(pop_size)]
    history = []

    for _ in range(gens):
        new_pop = []
        for _ in range(pop_size):
            p1 = selection(pop)
            p2 = selection(pop)
            child = mutation(crossover(p1, p2, c_rate), m_rate)
            new_pop.append(child)
        pop = new_pop
        best = min(pop, key=fitness)
        history.append(fitness(best))

    return min(pop, key=fitness), history

# ==============================
# PARETO FUNCTIONS
# ==============================
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def pareto_front(pop):
    front = []
    for p in pop:
        if not any(dominates(multi_objective_fitness(q), multi_objective_fitness(p)) for q in pop):
            front.append(p)
    return front

# ==============================
# MULTI-OBJECTIVE GA
# ==============================
def genetic_algorithm_multi(pop_size, gens, m_rate, c_rate):
    pop = [create_chromosome() for _ in range(pop_size)]
    pareto_sizes = []

    for _ in range(gens):
        front = pareto_front(pop)
        pareto_sizes.append(len(front))
        pop = [mutation(crossover(random.choice(front), random.choice(front), c_rate), m_rate)
               for _ in range(pop_size)]

    return pareto_front(pop)[0], pareto_sizes

# ==============================
# METRICS
# ==============================
def compute_metrics(solution, runtime):
    capacity_v = 0
    wasted = 0
    schedule = {}

    for e, (ts, r) in solution.items():
        schedule.setdefault((ts, r), []).append(e)

    for (ts, r), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)
        if students > room_capacity[r]:
            capacity_v += 1
        wasted += max(room_capacity[r] - students, 0)

    raw = fitness(solution)
    final_cost = round(raw / 1000, 1)

    return final_cost, capacity_v, wasted, raw, runtime

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config("Exam Scheduling GA", layout="wide")
st.title("🎓 University Exam Scheduling using Genetic Algorithm")

st.sidebar.header("Optimization Mode")
mode = st.sidebar.radio("Select Mode", ["Single Objective", "Multi Objective"])

st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 20, 200, 50)
gens = st.sidebar.slider("Generations", 50, 500, 100)
m_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
c_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)

if st.button("🚀 Run Genetic Algorithm"):
    start = time.time()

    if mode == "Single Objective":
        best, history = genetic_algorithm(pop_size, gens, m_rate, c_rate)
    else:
        best, pareto_sizes = genetic_algorithm_multi(pop_size, gens, m_rate, c_rate)
        history = pareto_sizes

    runtime = time.time() - start

    final_cost, cap_v, wasted, raw, runtime = compute_metrics(best, runtime)

    # ==============================
    # METRICS
    # ==============================
    st.subheader("📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Cost", final_cost)
    c2.metric("Capacity Violations", cap_v)
    c3.metric("Wasted Capacity", wasted)
    c4.metric("Computation Time (s)", round(runtime, 2))

    # ==============================
    # CONVERGENCE / PARETO PLOT
    # ==============================
    st.subheader("📈 Optimization Progress")
    fig, ax = plt.subplots()

    if mode == "Single Objective":
        ax.plot(history)
        ax.set_ylabel("Fitness")
    else:
        ax.plot(history)
        ax.set_ylabel("Pareto Front Size")

    ax.set_xlabel("Generation")
    st.pyplot(fig)

    # ==============================
    # TIMETABLE
    # ==============================
    st.subheader("🗓 Optimized Exam Timetable")
    df = pd.DataFrame([
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
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("**Course:** JIE42903 – Evolutionary Computing  \n**Method:** Genetic Algorithm")
