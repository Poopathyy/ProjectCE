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

def fitness(chromosome):
    penalty = 0
    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in room_usage.items():

        # Room–timeslot conflict (hard)
        if len(exams_in_room) > 1:
            penalty += 5 * (len(exams_in_room) - 1)

        students = sum(num_students_map[e] for e in exams_in_room)

        # Capacity violation (hard)
        if students > room_capacity[room]:
            penalty += 5

        # Room-type compatibility
        for e in exams_in_room:
            if exam_type_map[e].lower() == "practical" and room_type_map[room].lower() != "lab":
                penalty += 5
            if exam_type_map[e].lower() == "theory" and room_type_map[room].lower() == "lab":
                penalty += 2

        # Wasted capacity (soft)
        penalty += (room_capacity[room] - students) * 0.01

    return round(penalty, 2)

def selection(population):
    return min(random.sample(population, 3), key=fitness)

def crossover(p1, p2, rate):
    if random.random() > rate:
        return p1.copy()
    return {
        e: p1[e] if random.random() < 0.5 else p2[e]
        for e in exam_ids
    }

def mutation(chromosome, rate):
    for e in exam_ids:
        if random.random() < rate:
            chromosome[e] = (random.choice(timeslots), random.choice(room_ids))
    return chromosome

def evaluate_metrics(chromosome):
    capacity_violations = 0
    wasted_capacity = 0
    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in room_usage.items():
        students = sum(num_students_map[e] for e in exams_in_room)
        if students > room_capacity[room]:
            capacity_violations += 1
        wasted_capacity += max(room_capacity[room] - students, 0)

    return capacity_violations, wasted_capacity

def genetic_algorithm(pop_size, gens, mut_rate, cross_rate):
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
        best = min(population, key=fitness)
        history.append(fitness(best))

    return best, history

# ==============================
# Streamlit UI
# ==============================
st.set_page_config("University Exam Scheduling – GA", layout="wide")
st.title("🎓 University Exam Scheduling using Genetic Algorithm")

st.write(
    "This system optimizes exam scheduling by assigning exams to suitable classrooms "
    "while satisfying hard constraints and minimizing wasted capacity."
)

# ==============================
# Dataset Overview
# ==============================
st.subheader("📂 Dataset Overview")
c1, c2 = st.columns(2)
c1.markdown("### 🏫 Classrooms Dataset")
c1.dataframe(rooms, use_container_width=True)
c2.markdown("### 📝 Exam Timeslot Dataset")
c2.dataframe(exams, use_container_width=True)

# ==============================
# Sidebar Controls
# ==============================
st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 50, 10)
generations = st.sidebar.slider("Generations", 50, 500, 100, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1, 0.01)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8, 0.05)

# ==============================
# Run GA
# ==============================
if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing exam timetable..."):
        best_solution, fitness_history = genetic_algorithm(
            population_size,
            generations,
            mutation_rate,
            crossover_rate
        )

    final_cost = fitness(best_solution)
    capacity_violations, wasted_capacity = evaluate_metrics(best_solution)

    # ==============================
    # Final Metrics
    # ==============================
    st.subheader("📌 Final Optimization Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Cost", final_cost)   # ✅ SINGLE DIGIT
    col2.metric("Capacity Violations", capacity_violations)
    col3.metric("Wasted Capacity", wasted_capacity)

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
    # Timetable Output
    # ==============================
    st.subheader("🗓️ Optimized Exam Timetable")
    timetable = pd.DataFrame([
        {
            "Course Code": course_code_map[e],
            "Exam Day": exam_day_map[e],
            "Time Slot": ts,
            "Room": r,
            "Building": building_map[r],
            "Students": num_students_map[e],
            "Capacity": room_capacity[r]
        }
        for e, (ts, r) in best_solution.items()
    ])
    st.dataframe(timetable, use_container_width=True)

# ==============================
# Footer
# ==============================
st.markdown(
    "---\n"
    "**Course:** JIE42903 – Evolutionary Computing  \n"
    "**Case Study:** University Exam Scheduling  \n"
    "**Algorithm:** Genetic Algorithm"
)
