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
# Prepare Variables
# ==============================
exam_ids = exams['exam_id'].tolist()
timeslots = exams['exam_time'].unique().tolist()
room_ids = rooms['room_number'].tolist()
room_capacity = dict(zip(rooms['room_number'], rooms['capacity']))

STUDENTS_PER_EXAM = 30  # assumption (state this in report)

# ==============================
# Genetic Algorithm Functions
# ==============================

def create_chromosome():
    """Create one timetable"""
    return {
        exam: (random.choice(timeslots), random.choice(room_ids))
        for exam in exam_ids
    }


def fitness(chromosome):
    """Penalty-based fitness function"""
    penalty = 0
    room_usage = {}

    for exam, (ts, room) in chromosome.items():
        room_usage.setdefault((ts, room), []).append(exam)

    for (ts, room), exams_in_room in room_usage.items():
        # Hard constraint: multiple exams in same room
        if len(exams_in_room) > 1:
            penalty += 1000 * (len(exams_in_room) - 1)

        # Hard constraint: room capacity
        students = len(exams_in_room) * STUDENTS_PER_EXAM
        if students > room_capacity[room]:
            penalty += 1000

        # Soft constraint: room underutilization
        penalty += max(room_capacity[room] - students, 0) * 0.1

    return penalty


def selection(population):
    """Tournament selection"""
    tournament = random.sample(population, 3)
    tournament.sort(key=fitness)
    return tournament[0]


def crossover(parent1, parent2):
    """Uniform crossover"""
    return {
        exam: parent1[exam] if random.random() < 0.5 else parent2[exam]
        for exam in exam_ids
    }


def mutation(chromosome, rate):
    """Random mutation"""
    for exam in exam_ids:
        if random.random() < rate:
            chromosome[exam] = (
                random.choice(timeslots),
                random.choice(room_ids)
            )
    return chromosome


def genetic_algorithm(pop_size, generations, mutation_rate):
    """Main GA loop"""
    population = [create_chromosome() for _ in range(pop_size)]
    best_fitness_history = []

    for _ in range(generations):
        new_population = []

        for _ in range(pop_size):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population
        best = min(population, key=fitness)
        best_fitness_history.append(fitness(best))

    return best, best_fitness_history


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="Exam Scheduling GA", layout="wide")

st.title("🎓 University Exam Scheduling using Genetic Algorithm")
st.write(
    "This application optimizes university exam timetables using a "
    "Genetic Algorithm while satisfying room capacity and scheduling constraints."
)

# ==============================
# Sidebar Controls
# ==============================
st.sidebar.header("GA Parameters")

population_size = st.sidebar.slider(
    "Population Size", min_value=20, max_value=200, value=50, step=10
)

generations = st.sidebar.slider(
    "Number of Generations", min_value=50, max_value=500, value=100, step=50
)

mutation_rate = st.sidebar.slider(
    "Mutation Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01
)

# ==============================
# Run Button
# ==============================
if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing exam timetable..."):
        best_solution, fitness_history = genetic_algorithm(
            population_size,
            generations,
            mutation_rate
        )

    st.success(f"✅ Best Fitness Score: {fitness(best_solution)}")

    # ==============================
    # Convergence Plot
    # ==============================
    st.subheader("📈 GA Convergence Curve")

    fig, ax = plt.subplots()
    ax.plot(fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness vs Generation")
    st.pyplot(fig)

    # ==============================
    # Display Timetable
    # ==============================
    st.subheader("🗓️ Optimized Exam Timetable")

    timetable = pd.DataFrame(
        [
            {"Exam ID": exam, "Time Slot": ts, "Room": room}
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

