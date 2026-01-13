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
# Normalize Column Names (SAFE)
# ==============================
exams.columns = exams.columns.str.lower()
rooms.columns = rooms.columns.str.lower()

# ==============================
# Prepare Exam Data
# ==============================
exam_ids = exams['exam_id'].tolist()
timeslots = exams['exam_time'].unique().tolist()

course_code_map = dict(zip(exams['exam_id'], exams.get('course_code', exams['exam_id'])))
exam_day_map = dict(zip(exams['exam_id'], exams.get('exam_day', exams['exam_time'])))
num_students_map = dict(zip(exams['exam_id'], exams.get('num_students', [30]*len(exams))))

# ==============================
# Prepare Room Data
# ==============================
room_ids = rooms['room_number'].tolist()
room_capacity = dict(zip(rooms['room_number'], rooms['capacity']))
building_map = dict(zip(rooms['room_number'], rooms.get('building_name', rooms['room_number'])))

# ==============================
# Genetic Algorithm Functions
# ==============================

def create_chromosome():
    return {
        exam: (random.choice(timeslots), random.choice(room_ids))
        for exam in exam_ids
    }


def fitness(chromosome):
    penalty = 0
    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in room_usage.items():
        # Hard constraint: one exam per room per timeslot
        if len(exams_in_room) > 1:
            penalty += 1000 * (len(exams_in_room) - 1)

        # Hard constraint: room capacity
        students = sum(num_students_map[e] for e in exams_in_room)
        if students > room_capacity[room]:
            penalty += 1000

        # Soft constraint: underutilization
        penalty += max(room_capacity[room] - students, 0) * 0.1

    return penalty

def fitness_multi(chromosome, w1, w2, w3):
    penalty = 0
    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in room_usage.items():
        # Objective 1: exam clashes
        if len(exams_in_room) > 1:
            penalty += w1 * (len(exams_in_room) - 1)

        # Objective 2: room capacity
        students = sum(num_students_map[e] for e in exams_in_room)
        if students > room_capacity[room]:
            penalty += w2

        # Objective 3: room wastage
        penalty += w3 * max(room_capacity[room] - students, 0)

    return penalty

def selection(population):
    tournament = random.sample(population, 3)
    tournament.sort(key=fitness)
    return tournament[0]

def crossover(parent1, parent2, crossover_rate):
    # No crossover → copy parent1
    if random.random() > crossover_rate:
        return parent1.copy()

    # Uniform crossover
    return {
        exam: parent1[exam] if random.random() < 0.5 else parent2[exam]
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

def evaluate_final_metrics(chromosome):
    capacity_violations = 0
    wasted_capacity = 0

    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in room_usage.items():
        students = sum(num_students_map[e] for e in exams_in_room)
        capacity = room_capacity[room]

        # Capacity violation (hard constraint)
        if students > capacity:
            capacity_violations += 1

        # Wasted capacity (soft constraint)
        wasted_capacity += max(capacity - students, 0)

    return capacity_violations, wasted_capacity



def genetic_algorithm(pop_size, generations, mutation_rate,
                      crossover_rate, mode, w1, w2, w3):
    population = [create_chromosome() for _ in range(pop_size)]
    best_fitness_history = []

    for _ in range(generations):
        new_population = []

        for _ in range(pop_size):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2, crossover_rate)
            child = mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

        if mode == "Multi Objective":
            best = min(population, key=lambda x: fitness_multi(x, w1, w2, w3))
            best_fitness_history.append(fitness_multi(best, w1, w2, w3))
        else:
            best = min(population, key=fitness)
            best_fitness_history.append(fitness(best))

    return best, best_fitness_history

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Exam Scheduling GA", layout="wide")

st.title("🎓 University Exam Scheduling using Genetic Algorithm")
st.write(
    "This application optimizes university exam timetables while considering "
    "course information, student numbers, room capacity, and building allocation."
)

# ==============================
# Datasets overview
# ==============================
st.subheader("📂 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏫 Classrooms Dataset")
    st.dataframe(
        rooms.head(10),
        use_container_width=True,
        height=350
    )

with col2:
    st.markdown("### 📝 Exam Timeslot Dataset")
    st.dataframe(
        exams.head(10),
        use_container_width=True,
        height=350
    )

# ==============================
# Sidebar Controls
# ==============================
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
    st.sidebar.markdown("### Multi-Objective Weights")

    w_clash = st.sidebar.slider(
        "Clash Penalty Weight", 1000, 10000, 3000, 500
    )

    w_capacity = st.sidebar.slider(
        "Capacity Penalty Weight", 1000, 10000, 3000, 500
    )

    w_wastage = st.sidebar.slider(
        "Room Wastage Weight", 1, 50, 10, 1
    )
else:
    # Default values (not used in single objective)
    w_clash = w_capacity = w_wastage = None

# ==============================
# Run Button
# ==============================
if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing exam timetable..."):
       best_solution, fitness_history = genetic_algorithm(
        population_size,
        generations,
        mutation_rate,
        crossover_rate,
        optimization_mode,
        w_clash,
        w_capacity,
        w_wastage
    )

    # ---- Best score ----
    if optimization_mode == "Multi Objective":
        best_score = fitness_multi(
            best_solution, w_clash, w_capacity, w_wastage
        )
    else:
        best_score = fitness(best_solution)

    # ---- FINAL METRICS (DEFINE FIRST) ----
    capacity_violations, wasted_capacity = evaluate_final_metrics(best_solution)

    # ---- DISPLAY METRICS ----
    st.subheader("📌 Final Optimization Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Cost", round(best_score, 2))
    col2.metric("Capacity Violations", capacity_violations)
    col3.metric("Wasted Capacity", wasted_capacity)

    # ---- CONVERGENCE PLOT ----
    st.subheader("📈 GA Convergence Curve")

    fig, ax = plt.subplots()
    ax.plot(fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"GA Convergence Curve ({optimization_mode})")
    st.pyplot(fig)

    # ---- TIMETABLE ----
    st.subheader("🗓️ Optimized Exam Timetable")

    timetable = pd.DataFrame(
        [
            {
                "Course Code": course_code_map[exam],
                "Exam ID": exam,
                "Exam Day": exam_day_map[exam],
                "Time Slot": ts,
                "Room": room,
                "Building": building_map[room],
                "No. of Students": num_students_map[exam],
                "Room Capacity": room_capacity[room]
            }
            for exam, (ts, room) in best_solution.items()
        ]
    )

    st.dataframe(timetable, use_container_width=True)

# ==============================
# Footer
# ==============================
st.markdown(
    "---\n"
    "**Course:** JIE42903 – Evolutionary Computing  \n"
    "**Case Study:** University Exam Scheduling  \n"
    "**Method:** Genetic Algorithm"
)
