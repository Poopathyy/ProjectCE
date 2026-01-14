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

# Normalize column names
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

    for exam, (ts, room) in solution.items():
        schedule.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)

        # Room–Timeslot conflict
        if len(exams_here) > 1:
            penalty += 1000 * (len(exams_here) - 1)

        # Capacity violation
        if students > room_capacity[room]:
            penalty += 1000

        # Room–type compatibility
        for e in exams_here:
            if exam_type[e].lower() == "practical" and "lab" not in room_type[room].lower():
                penalty += 500
            if exam_type[e].lower() == "theory" and "lab" in room_type[room].lower():
                penalty += 300

        # Wasted capacity (soft constraint)
        penalty += max(room_capacity[room] - students, 0) * 0.1

    return penalty


def selection(population):
    return min(random.sample(population, 3), key=fitness)


def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1.copy()
    return {e: p1[e] if random.random() < 0.5 else p2[e] for e in exam_ids}


def mutation(chromosome, rate):
    for e in exam_ids:
        if random.random() < rate:
            chromosome[e] = (random.choice(timeslots), random.choice(room_ids))
    return chromosome


def genetic_algorithm(pop_size, generations, mutation_rate, crossover_rate):
    population = [create_chromosome() for _ in range(pop_size)]
    history = []

    for _ in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2, crossover_rate)
            child = mutation(child, mutation_rate)
            new_pop.append(child)
        population = new_pop

        best = min(population, key=fitness)
        history.append(fitness(best))

    return best, history


# ==============================
# Metrics
# ==============================
def compute_metrics(solution, runtime):
    capacity_violations = 0
    wasted_capacity = 0
    schedule = {}

    for exam, (ts, room) in solution.items():
        schedule.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_here in schedule.items():
        students = sum(num_students[e] for e in exams_here)
        if students > room_capacity[room]:
            capacity_violations += 1
        wasted_capacity += max(room_capacity[room] - students, 0)

    raw_fitness = fitness(solution)

    # Convert raw fitness to ONE digit
    final_cost = round(raw_fitness / 1000, 1)

    return final_cost, capacity_violations, wasted_capacity, raw_fitness, runtime


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

if st.button("🚀 Run Genetic Algorithm"):
    start_time = time.time()

    best_solution, fitness_history = genetic_algorithm(
        population_size,
        generations,
        mutation_rate,
        crossover_rate
    )

    runtime = time.time() - start_time

    final_cost, capacity_violations, wasted_capacity, raw_fitness, runtime = compute_metrics(
        best_solution, runtime
    )

    # ==============================
    # Metrics Display
    # ==============================
    st.subheader("📊 Performance Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Cost", final_cost)
    c2.metric("Capacity Violations", capacity_violations)
    c3.metric("Wasted Capacity", wasted_capacity)
    c4.metric("Computation Time (s)", round(runtime, 2))

    # ==============================
    # Convergence Plot
    # ==============================
    st.subheader("📈 GA Convergence Curve")
    fig, ax = plt.subplots()
    ax.plot(fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    # ==============================
    # Timetable
    # ==============================
    st.subheader("🗓️ Optimized Exam Timetable")
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
        for e, (ts, r) in best_solution.items()
    ])

    st.dataframe(timetable, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Course:** JIE42903 – Evolutionary Computing  \n**Method:** Genetic Algorithm")
