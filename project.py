import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    exams = pd.read_csv("exam_timeslot.csv")
    rooms = pd.read_csv("classrooms.csv")
    return exams, rooms

exams, rooms = load_data()

exam_ids = exams['exam_id'].tolist()
timeslots = exams['exam_time'].unique().tolist()
room_ids = rooms['room_number'].tolist()
room_capacity = dict(zip(rooms['room_number'], rooms['capacity']))

STUDENTS_PER_EXAM = 30  # assumption

# -------------------------------
# Genetic Algorithm Components
# -------------------------------
def create_chromosome():
    return {
        exam: (random.choice(timeslots), random.choice(room_ids))
        for exam in exam_ids
    }


def fitness_multi(chromosome, w_clash, w_capacity, w_wastage):
    clashes = 0
    over_capacity = 0
    wastage = 0

    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in room_usage.items():
        # Exam clashes
        if len(exams_in_room) > 1:
            clashes += len(exams_in_room) - 1

        capacity = room_capacity[room]
        used = len(exams_in_room) * STUDENTS_PER_EXAM

        # Over capacity
        if used > capacity:
            over_capacity += (used - capacity)

        # Wastage
        wastage += max(capacity - used, 0)

    return (
        w_clash * clashes +
        w_capacity * over_capacity +
        w_wastage * wastage
    )


def selection(population, w1, w2, w3):
    tournament = random.sample(population, 3)
    tournament.sort(key=lambda x: fitness_multi(x, w1, w2, w3))
    return tournament[0]


def crossover(p1, p2):
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


def genetic_algorithm(pop_size, generations, mutation_rate, w1, w2, w3):
    population = [create_chromosome() for _ in range(pop_size)]
    best_history = []

    for _ in range(generations):
        new_population = []

        for _ in range(pop_size):
            p1 = selection(population, w1, w2, w3)
            p2 = selection(population, w1, w2, w3)
            child = crossover(p1, p2)
            child = mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population
        best = min(
            population,
            key=lambda x: fitness_multi(x, w1, w2, w3)
        )
        best_history.append(
            fitness_multi(best, w1, w2, w3)
        )

    return best, best_history

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📘 University Exam Scheduling using Multi-Objective GA")
st.write(
    "This application optimizes university exam timetables using a "
    "Genetic Algorithm with **multi-objective optimization**."
)

# Sidebar Controls
st.sidebar.header("GA Parameters")

population_size = st.sidebar.slider(
    "Population Size", 20, 200, 50
)
generations = st.sidebar.slider(
    "Generations", 50, 500, 100
)
mutation_rate = st.sidebar.slider(
    "Mutation Rate", 0.01, 0.5, 0.1
)

st.sidebar.header("Objective Weights")

w_clash = st.sidebar.slider(
    "Weight: Exam Clashes (Hard Constraint)",
    500, 5000, 2000
)
w_capacity = st.sidebar.slider(
    "Weight: Room Overcapacity",
    1, 100, 10
)
w_wastage = st.sidebar.slider(
    "Weight: Room Wastage",
    1, 50, 5
)

# Run Button
if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing exam timetable..."):
        best_solution, fitness_history = genetic_algorithm(
            population_size,
            generations,
            mutation_rate,
            w_clash,
            w_capacity,
            w_wastage
        )

    best_score = fitness_multi(
        best_solution,
        w_clash,
        w_capacity,
        w_wastage
    )

    st.success(f"✅ Best Fitness Score: {best_score}")

    # Convergence Plot
    fig, ax = plt.subplots()
    ax.plot(fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Convergence Curve (Multi-Objective)")
    st.pyplot(fig)

    # Timetable Output
    timetable = pd.DataFrame([
        {
            "Exam": exam,
            "Timeslot": ts,
            "Room": room
        }
        for exam, (ts, room) in best_solution.items()
    ])

    st.subheader("📅 Optimized Exam Timetable")
    st.dataframe(timetable)

    st.markdown("### 🔍 Objective Interpretation")
    st.write(
        "- **Exam Clashes** are heavily penalized to ensure feasibility.\n"
        "- **Room Overcapacity** is minimized to respect physical constraints.\n"
        "- **Room Wastage** encourages efficient resource utilization.\n\n"
        "Adjusting weights allows exploration of trade-offs between objectives."
    )
