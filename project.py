import streamlit as st
import pandas as pd
import random
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

# ==============================
# Normalize Column Names
# ==============================
exams.columns = exams.columns.str.lower()
rooms.columns = rooms.columns.str.lower()

# ==============================
# Prepare Exam Data
# ==============================
exam_ids = exams['exam_id'].tolist()
timeslots = exams['exam_time'].unique().tolist()

course_code_map = dict(zip(exams['exam_id'], exams['course_code']))
exam_day_map = dict(zip(exams['exam_id'], exams['exam_day']))
num_students_map = dict(zip(exams['exam_id'], exams['num_students']))
exam_type_map = dict(zip(exams['exam_id'], exams['exam_type']))

# ==============================
# Prepare Room Data
# ==============================
room_ids = rooms['room_number'].tolist()
room_capacity = dict(zip(rooms['room_number'], rooms['capacity']))
building_map = dict(zip(rooms['room_number'], rooms['building_name']))
room_type_map = dict(zip(rooms['room_number'], rooms['room_type']))

# ==============================
# Genetic Algorithm Functions
# ==============================
def create_chromosome():
    return {
        exam: (random.choice(timeslots), random.choice(room_ids))
        for exam in exam_ids
    }

# -------- SINGLE OBJECTIVE FITNESS --------
def fitness(chromosome):
    penalty = 0
    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

        exam_type = exam_type_map[exam].lower()
        room_type = room_type_map[room].lower()

        # HARD: Room-Type Compatibility
        if exam_type == "practical" and room_type != "lab":
            penalty += 2000
        if exam_type == "theory" and room_type == "lab":
            penalty += 2000

    for (ts, room), exams_in_room in room_usage.items():
        # HARD: Room–Timeslot Conflict
        if len(exams_in_room) > 1:
            penalty += 1000 * (len(exams_in_room) - 1)

        students = sum(num_students_map[e] for e in exams_in_room)

        # HARD: Capacity
        if students > room_capacity[room]:
            penalty += 1000

        # SOFT: Wasted Capacity
        penalty += max(room_capacity[room] - students, 0) * 0.1

    return penalty

# -------- MULTI OBJECTIVE FITNESS --------
def fitness_multi(chromosome, w1, w2, w3):
    penalty = 0
    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

        exam_type = exam_type_map[exam].lower()
        room_type = room_type_map[room].lower()

        # Objective 1: Room-Type Compatibility
        if exam_type == "practical" and room_type != "lab":
            penalty += w1
        if exam_type == "theory" and room_type == "lab":
            penalty += w1

    for (ts, room), exams_in_room in room_usage.items():
        # Objective 2: Room Conflict
        if len(exams_in_room) > 1:
            penalty += w2 * (len(exams_in_room) - 1)

        students = sum(num_students_map[e] for e in exams_in_room)

        # Objective 3: Capacity
        if students > room_capacity[room]:
            penalty += w2

        # Objective 4: Wastage
        penalty += w3 * max(room_capacity[room] - students, 0)

    return penalty

# -------- GA OPERATORS --------
def selection(population):
    tournament = random.sample(population, 3)
    tournament.sort(key=fitness)
    return tournament[0]

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

# -------- FINAL METRICS --------
def evaluate_final_metrics(chromosome):
    capacity_violations = 0
    wasted_capacity = 0
    room_type_violations = 0
    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

        exam_type = exam_type_map[exam].lower()
        room_type = room_type_map[room].lower()

        if exam_type == "practical" and room_type != "lab":
            room_type_violations += 1
        if exam_type == "theory" and room_type == "lab":
            room_type_violations += 1

    for (ts, room), exams_in_room in room_usage.items():
        students = sum(num_students_map[e] for e in exams_in_room)
        if students > room_capacity[room]:
            capacity_violations += 1
        wasted_capacity += max(room_capacity[room] - students, 0)

    return capacity_violations, wasted_capacity, room_type_violations

# -------- MAIN GA --------
def genetic_algorithm(pop_size, generations, mutation_rate, crossover_rate,
                      mode, w1, w2, w3):

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

        if mode == "Multi Objective":
            best = min(population, key=lambda x: fitness_multi(x, w1, w2, w3))
            history.append(fitness_multi(best, w1, w2, w3))
        else:
            best = min(population, key=fitness)
            history.append(fitness(best))

    return best, history

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Exam Scheduling GA", layout="wide")
st.title("🎓 University Exam Scheduling using Genetic Algorithm")

# Sidebar
st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 50, 10)
generations = st.sidebar.slider("Generations", 50, 500, 100, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1, 0.01)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8, 0.05)

optimization_mode = st.sidebar.radio(
    "Optimization Mode",
    ["Single Objective", "Multi Objective"]
)

if optimization_mode == "Multi Objective":
    w_clash = st.sidebar.slider("Clash / Type Weight", 1000, 10000, 3000, 500)
    w_capacity = st.sidebar.slider("Capacity Weight", 1000, 10000, 3000, 500)
    w_wastage = st.sidebar.slider("Wastage Weight", 1, 50, 10, 1)
else:
    w_clash = w_capacity = w_wastage = None

# Run
if st.button("🚀 Run Genetic Algorithm"):
    best, history = genetic_algorithm(
        population_size, generations,
        mutation_rate, crossover_rate,
        optimization_mode,
        w_clash, w_capacity, w_wastage
    )

    if optimization_mode == "Multi Objective":
        best_score = fitness_multi(best, w_clash, w_capacity, w_wastage)
    else:
        best_score = fitness(best)

    cap_v, waste, type_v = evaluate_final_metrics(best)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Cost", round(best_score, 2))
    col2.metric("Capacity Violations", cap_v)
    col3.metric("Wasted Capacity", waste)
    col4.metric("Room-Type Violations", type_v)

    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_title("GA Convergence Curve")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    st.subheader("🗓️ Optimized Exam Timetable")
    timetable = pd.DataFrame([
        {
            "Course Code": course_code_map[e],
            "Exam Day": exam_day_map[e],
            "Time Slot": ts,
            "Room": r,
            "Room Type": room_type_map[r],
            "Building": building_map[r],
            "Students": num_students_map[e],
            "Capacity": room_capacity[r]
        }
        for e, (ts, r) in best.items()
    ])
    st.dataframe(timetable, use_container_width=True)
