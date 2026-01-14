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
# GA Helpers
# ==============================
def create_chromosome():
    return {e: (random.choice(timeslots), random.choice(room_ids)) for e in exam_ids}

# ---------- SINGLE OBJECTIVE FITNESS ----------
def fitness(solution):
    penalty = 0
    schedule = {}

    for exam, (ts, room) in solution.items():
        schedule.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)

        if len(exams_here) > 1:
            penalty += 1000 * (len(exams_here) - 1)

        if students > room_capacity[room]:
            penalty += 1000

        for e in exams_here:
            if exam_type[e] == "practical" and "lab" not in room_type[room]:
                penalty += 500
            if exam_type[e] == "theory" and "lab" in room_type[room]:
                penalty += 300

        penalty += max(room_capacity[room] - students, 0) * 0.1

    return penalty

# ---------- MULTI OBJECTIVE FITNESS ----------
def multi_objective_fitness(solution):
    cap_viol = 0
    conflicts = 0
    wasted = 0
    schedule = {}

    for exam, (ts, room) in solution.items():
        schedule.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)

        if students > room_capacity[room]:
            cap_viol += 1

        if len(exams_here) > 1:
            conflicts += len(exams_here) - 1

        for e in exams_here:
            if exam_type[e] == "practical" and "lab" not in room_type[room]:
                conflicts += 1
            if exam_type[e] == "theory" and "lab" in room_type[room]:
                conflicts += 1

        wasted += max(room_capacity[room] - students, 0)

    return cap_viol, conflicts, wasted

def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def pareto_front(pop):
    front = []
    for p in pop:
        if not any(dominates(multi_objective_fitness(q), multi_objective_fitness(p)) for q in pop):
            front.append(p)
    return front

# ==============================
# GA Core
# ==============================
def selection(pop):
    return min(random.sample(pop, 3), key=fitness)

def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1.copy()
    return {e: p1[e] if random.random() < 0.5 else p2[e] for e in exam_ids}

def mutation(chrom, rate):
    for e in exam_ids:
        if random.random() < rate:
            chrom[e] = (random.choice(timeslots), random.choice(room_ids))
    return chrom

def genetic_algorithm(pop_size, gens, m_rate, c_rate):
    pop = [create_chromosome() for _ in range(pop_size)]
    history = []

    for _ in range(gens):
        pop = [mutation(crossover(selection(pop), selection(pop), c_rate), m_rate)
               for _ in range(pop_size)]
        best = min(pop, key=fitness)
        history.append(fitness(best))

    return best, history

def genetic_algorithm_multi(pop_size, gens, m_rate, c_rate):
    pop = [create_chromosome() for _ in range(pop_size)]

    for _ in range(gens):
        front = pareto_front(pop)
        pop = [mutation(crossover(random.choice(front), random.choice(front), c_rate), m_rate)
               for _ in range(pop_size)]

    return pareto_front(pop)[0]

# ==============================
# Streamlit UI
# ==============================
st.set_page_config("Exam Scheduling GA", layout="wide")
st.title("🎓 University Exam Scheduling using Genetic Algorithm")

st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Generations", 50, 500, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)

optimization_mode = st.sidebar.radio(
    "Optimization Mode",
    ["Single Objective", "Multi Objective"]
)

if st.button("🚀 Run Genetic Algorithm"):
    start = time.time()

    if optimization_mode == "Single Objective":
        best, history = genetic_algorithm(
            population_size, generations, mutation_rate, crossover_rate
        )
        raw_fitness = fitness(best)

    else:
        best = genetic_algorithm_multi(
            population_size, generations, mutation_rate, crossover_rate
        )
        c, k, w = multi_objective_fitness(best)
        raw_fitness = c * 1000 + k * 500 + w * 0.1
        history = None

    runtime = time.time() - start
    final_cost = round(raw_fitness / 1000, 1)

    # ==============================
    # Metrics
    # ==============================
    cap_v = 0
    waste = 0
    sched = {}

    for e, (t, r) in best.items():
        sched.setdefault((t, r), []).append(e)

    for (t, r), exs in sched.items():
        s = sum(num_students[e] for e in exs)
        if s > room_capacity[r]:
            cap_v += 1
        waste += max(room_capacity[r] - s, 0)

    st.subheader("📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Cost", final_cost)
    c2.metric("Capacity Violations", cap_v)
    c3.metric("Wasted Capacity", waste)
    c4.metric("Computation Time (s)", round(runtime, 2))

    # ==============================
    # Convergence
    # ==============================
    if history:
        st.subheader("📈 GA Convergence Curve")
        fig, ax = plt.subplots()
        ax.plot(history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        st.pyplot(fig)
    else:
        st.info("Multi-objective GA uses Pareto dominance (no single convergence curve).")

    # ==============================
    # Timetable
    # ==============================
    st.subheader("🗓️ Optimized Exam Timetable")
    df = pd.DataFrame([
        {
            "Exam ID": e,
            "Exam Type": exam_type[e],
            "Timeslot": t,
            "Room": r,
            "Room Type": room_type[r],
            "Students": num_students[e],
            "Capacity": room_capacity[r]
        }
        for e, (t, r) in best.items()
    ])
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("**Course:** JIE42903 – Evolutionary Computing  \n**Method:** Genetic Algorithm")
