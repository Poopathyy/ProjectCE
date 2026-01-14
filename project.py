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
room_ids = rooms["room_number"].tolist()

num_students = dict(zip(exams["exam_id"], exams["num_students"]))
exam_type = dict(zip(exams["exam_id"], exams["exam_type"].str.lower()))
room_capacity = dict(zip(rooms["room_number"], rooms["capacity"]))
room_type = dict(zip(rooms["room_number"], rooms["room_type"].str.lower()))

# ==============================
# GA Components
# ==============================
def create_chromosome():
    return {e: (random.choice(timeslots), random.choice(room_ids)) for e in exam_ids}

def fitness(chromosome):
    penalty = 0
    schedule = {}

    for e, (t, r) in chromosome.items():
        schedule.setdefault((t, r), []).append(e)

    for (t, r), exams_here in schedule.items():
        # Room–timeslot conflict
        if len(exams_here) > 1:
            penalty += 10 * (len(exams_here) - 1)

        students = sum(num_students[e] for e in exams_here)

        # Capacity violation
        if students > room_capacity[r]:
            penalty += 5

        # Room-type compatibility
        for e in exams_here:
            if exam_type[e] == "practical" and "lab" not in room_type[r]:
                penalty += 3
            if exam_type[e] == "theory" and "lab" in room_type[r]:
                penalty += 2

        # Wasted capacity
        penalty += max(room_capacity[r] - students, 0) * 0.05

    return penalty

def selection(pop):
    return min(random.sample(pop, 3), key=fitness)

def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1.copy()
    return {e: p1[e] if random.random() < 0.5 else p2[e] for e in exam_ids}

def mutation(ch, rate):
    for e in exam_ids:
        if random.random() < rate:
            ch[e] = (random.choice(timeslots), random.choice(room_ids))
    return ch

def evaluate_metrics(ch):
    capacity_violations = 0
    wasted_capacity = 0
    schedule = {}

    for e, (t, r) in ch.items():
        schedule.setdefault((t, r), []).append(e)

    for (t, r), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)
        if students > room_capacity[r]:
            capacity_violations += 1
        wasted_capacity += max(room_capacity[r] - students, 0)

    return capacity_violations, wasted_capacity

def genetic_algorithm(pop_size, gens, mut_rate, cross_rate):
    start = time.time()
    population = [create_chromosome() for _ in range(pop_size)]
    history = []

    for _ in range(gens):
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = selection(population), selection(population)
            child = mutation(crossover(p1, p2, cross_rate), mut_rate)
            new_pop.append(child)
        population = new_pop
        best = min(population, key=fitness)
        history.append(fitness(best))

    runtime = time.time() - start
    return best, history, runtime

# ==============================
# Streamlit UI
# ==============================
st.set_page_config("Exam Scheduling GA", layout="wide")
st.title("🎓 University Exam Scheduling using Genetic Algorithm")

# Sidebar
st.sidebar.header("GA Parameters")
pop_size = st.sidebar.slider("Population Size", 20, 200, 50, 10)
gens = st.sidebar.slider("Generations", 50, 500, 100, 50)
mut_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1, 0.01)
cross_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8, 0.05)

# Dataset View
st.subheader("📂 Dataset Overview")
c1, c2 = st.columns(2)
c1.dataframe(rooms, use_container_width=True)
c2.dataframe(exams, use_container_width=True)

# Run GA
if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing..."):
        best, history, runtime = genetic_algorithm(pop_size, gens, mut_rate, cross_rate)

    raw_fitness = fitness(best)
    final_cost = round(raw_fitness / (raw_fitness + 1), 1)
    cap_vio, waste = evaluate_metrics(best)

    # Metrics
    st.subheader("📌 Final Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw Fitness", round(raw_fitness, 2))
    m2.metric("Final Cost", final_cost)
    m3.metric("Capacity Violations", cap_vio)
    m4.metric("Computation Time (s)", round(runtime, 2))

    st.metric("Wasted Capacity", waste)

    # Convergence
    st.subheader("📈 Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    # Timetable
    st.subheader("🗓 Optimized Timetable")
    timetable = pd.DataFrame([
        {
            "Exam ID": e,
            "Exam Type": exam_type[e],
            "Timeslot": best[e][0],
            "Room": best[e][1],
            "Room Type": room_type[best[e][1]],
            "Students": num_students[e],
            "Capacity": room_capacity[best[e][1]]
        }
        for e in exam_ids
    ])
    st.dataframe(timetable, use_container_width=True)

# Footer
st.markdown(
    "---\n"
    "**Course:** JIE42903 – Evolutionary Computing  \n"
    "**Case Study:** University Exam Scheduling  \n"
    "**Algorithm:** Genetic Algorithm"
)
