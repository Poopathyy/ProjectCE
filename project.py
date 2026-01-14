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
# GA FUNCTIONS
# ==============================
def create_chromosome():
    return {e: (random.choice(timeslots), random.choice(room_ids)) for e in exam_ids}

# -------- Single Objective Fitness --------
def fitness_single(solution):
    penalty = 0
    usage = {}

    for e, (ts, r) in solution.items():
        usage.setdefault((ts, r), []).append(e)

    for (ts, r), exams_here in usage.items():
        students = sum(num_students[e] for e in exams_here)

        if len(exams_here) > 1:
            penalty += 1000 * (len(exams_here) - 1)

        if students > room_capacity[r]:
            penalty += 1000

        for e in exams_here:
            if exam_type[e] == "practical" and "lab" not in room_type[r].lower():
                penalty += 500
            if exam_type[e] == "theory" and "lab" in room_type[r].lower():
                penalty += 300

        penalty += max(room_capacity[r] - students, 0) * 0.1

    return penalty

# -------- Multi Objective Fitness --------
def fitness_multi(solution):
    hard_penalty = 0
    wasted_capacity = 0
    usage = {}

    for e, (ts, r) in solution.items():
        usage.setdefault((ts, r), []).append(e)

    for (ts, r), exams_here in usage.items():
        students = sum(num_students[e] for e in exams_here)

        if len(exams_here) > 1:
            hard_penalty += 1

        if students > room_capacity[r]:
            hard_penalty += 1

        for e in exams_here:
            if exam_type[e] == "practical" and "lab" not in room_type[r].lower():
                hard_penalty += 1
            if exam_type[e] == "theory" and "lab" in room_type[r].lower():
                hard_penalty += 1

        wasted_capacity += max(room_capacity[r] - students, 0)

    return hard_penalty, wasted_capacity

# -------- GA Core --------
def selection(pop):
    return min(random.sample(pop, 3), key=lambda x: fitness_single(x))

def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1.copy()
    return {e: p1[e] if random.random() < 0.5 else p2[e] for e in exam_ids}

def mutation(ch, rate):
    for e in exam_ids:
        if random.random() < rate:
            ch[e] = (random.choice(timeslots), random.choice(room_ids))
    return ch

def genetic_algorithm(pop_size, gens, mut_rate, cross_rate, mode):
    population = [create_chromosome() for _ in range(pop_size)]
    history = []

    for _ in range(gens):
        new_pop = []
        for _ in range(pop_size):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2, cross_rate)
            child = mutation(child, mut_rate)
            new_pop.append(child)
        population = new_pop

        if mode == "Single Objective":
            best = min(population, key=fitness_single)
            history.append(fitness_single(best))
        else:
            best = min(population, key=lambda x: sum(fitness_multi(x)))
            history.append(sum(fitness_multi(best)))

    return best, history

# ==============================
# METRICS
# ==============================
def compute_metrics(solution, runtime):
    cap_viol = 0
    waste = 0
    usage = {}

    for e, (ts, r) in solution.items():
        usage.setdefault((ts, r), []).append(e)

    for (ts, r), exams_here in usage.items():
        students = sum(num_students[e] for e in exams_here)
        if students > room_capacity[r]:
            cap_viol += 1
        waste += max(room_capacity[r] - students, 0)

    raw = fitness_single(solution)
    final_cost = round(raw / 1000, 1)

    return final_cost, cap_viol, waste, raw, runtime

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config("Exam Scheduling GA", layout="wide")
st.title("🎓 University Exam Scheduling using Genetic Algorithm")

tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Results", "🗓️ Timetable"])

with tab1:
    st.sidebar.header("GA Parameters")
    pop_size = st.sidebar.slider("Population Size", 20, 200, 50)
    gens = st.sidebar.slider("Generations", 50, 500, 100)
    mut = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
    cross = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)

    mode = st.radio("Optimization Mode", ["Single Objective", "Multi Objective"])

    if st.button("🚀 Run Genetic Algorithm"):
        start = time.time()
        best_solution, history = genetic_algorithm(pop_size, gens, mut, cross, mode)
        runtime = time.time() - start
        st.session_state["result"] = (best_solution, history, runtime, mode)

with tab2:
    if "result" in st.session_state:
        sol, hist, runtime, mode = st.session_state["result"]
        final_cost, cap_v, waste, raw, _ = compute_metrics(sol, runtime)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Cost", final_cost)
        c2.metric("Capacity Violations", cap_v)
        c3.metric("Wasted Capacity", waste)
        c4.metric("Computation Time (s)", round(runtime, 2))

        st.subheader("📈 GA Convergence Curve")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(hist)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title(f"GA Convergence ({mode})")
        st.pyplot(fig)

with tab3:
    if "result" in st.session_state:
        sol = st.session_state["result"][0]
        table = pd.DataFrame([
            {
                "Exam ID": e,
                "Type": exam_type[e],
                "Timeslot": ts,
                "Room": r,
                "Room Type": room_type[r],
                "Students": num_students[e],
                "Capacity": room_capacity[r]
            }
            for e, (ts, r) in sol.items()
        ])
        st.dataframe(table, use_container_width=True)

st.markdown(
    "---\n"
    "**Course:** JIE42903 – Evolutionary Computing  \n"
    "**Method:** Genetic Algorithm"
)
