import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# -----------------------------
# Load datasets
# -----------------------------
@st.cache_data
def load_data():
    rooms = pd.read_csv("classrooms.csv")
    exams = pd.read_csv("exam_timeslot.csv")
    return rooms, exams

rooms_df, exams_df = load_data()
NUM_EXAMS = len(exams_df)
ROOM_IDS = rooms_df["classroom_id"].tolist()

# -----------------------------
# Multi-Objective Fitness Function
# -----------------------------
def fitness_function_multi(chromosome):
    hard_penalty = 0  # capacity + room-time clash
    soft_penalty = 0  # room type mismatch + high utilization

    # Exam-room penalties
    for i, exam in exams_df.iterrows():
        room = rooms_df[rooms_df["classroom_id"] == chromosome[i]].iloc[0]

        # Hard constraints
        if exam["num_students"] > room["capacity"]:
            hard_penalty += (exam["num_students"] - room["capacity"]) * 10

        for j in range(i + 1, NUM_EXAMS):
            if chromosome[i] == chromosome[j] and exams_df.iloc[i]["exam_day"] == exams_df.iloc[j]["exam_day"] and exams_df.iloc[i]["exam_time"] == exams_df.iloc[j]["exam_time"]:
                hard_penalty += 50

        # Soft constraints
        if exam["exam_type"] == "Practical" and room["room_type"] != "Lab":
            soft_penalty += 5
        if exam["num_students"] > 0.8 * room["capacity"]:
            soft_penalty += 3

    return [hard_penalty, soft_penalty]

# -----------------------------
# GA Components
# -----------------------------
def create_individual():
    return [random.choice(ROOM_IDS) for _ in range(NUM_EXAMS)]

def crossover(parent1, parent2):
    point = random.randint(1, NUM_EXAMS - 2)
    return parent1[:point] + parent2[point:]

def mutate(individual, mutation_rate):
    for i in range(NUM_EXAMS):
        if random.random() < mutation_rate:
            individual[i] = random.choice(ROOM_IDS)
    return individual

def pareto_selection(population, fitnesses, k=3):
    """Tournament selection based on Pareto dominance."""
    selected = random.sample(list(zip(population, fitnesses)), k)
    # Sort by non-domination: lower hard_penalty is prioritized
    return min(selected, key=lambda x: (x[1][0], x[1][1]))[0]

# -----------------------------
# Multi-Objective GA Execution
# -----------------------------
def run_moga(pop_size, generations, mutation_rate):
    population = [create_individual() for _ in range(pop_size)]
    pareto_front = []

    for gen in range(generations):
        fitnesses = [fitness_function_multi(ind) for ind in population]

        # Build Pareto front
        pareto_front_gen = []
        for i, fit_i in enumerate(fitnesses):
            dominated = False
            for j, fit_j in enumerate(fitnesses):
                if j != i:
                    # j dominates i if all objectives are <= and at least one is <
                    if all(fj <= fi for fi, fj in zip(fit_i, fit_j)) and any(fj < fi for fi, fj in zip(fit_i, fit_j)):
                        dominated = True
                        break
            if not dominated:
                pareto_front_gen.append((population[i], fit_i))
        pareto_front = pareto_front_gen  # update Pareto front

        # Create next generation
        new_population = []
        for _ in range(pop_size):
            parent1 = pareto_selection(population, fitnesses)
            parent2 = pareto_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    return pareto_front

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Multi-Objective Exam Scheduling GA", layout="wide")
st.title("📅 Multi-Objective University Exam Scheduling (GA)")

# Sidebar - GA parameters
st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 80)
generations = st.sidebar.slider("Generations", 50, 500, 200)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

if st.sidebar.button("Run Multi-Objective GA"):
    with st.spinner("Optimizing exam schedule..."):
        pareto_front = run_moga(population_size, generations, mutation_rate)

    st.success("Optimization Completed!")

    # -----------------------------
    # Pareto Front Plot
    # -----------------------------
    st.subheader("📊 Pareto Front (Hard vs Soft Penalties)")
    hard_penalties = [pf[1][0] for pf in pareto_front]
    soft_penalties = [pf[1][1] for pf in pareto_front]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(hard_penalties, soft_penalties, color='red')
    ax.set_xlabel("Hard Constraint Penalty")
    ax.set_ylabel("Soft Constraint Penalty")
    ax.set_title("Pareto Front: Trade-off Between Objectives")
    st.pyplot(fig)

    # -----------------------------
    # Display Example Schedule from Pareto Front
    # -----------------------------
    st.subheader("🗓️ Example Exam Timetable from Pareto Front")
    # Select the solution with minimum hard penalty
    best_solution = min(pareto_front, key=lambda x: x[1][0])[0]
    timetable = exams_df.copy()
    timetable["room_id"] = best_solution
    timetable = timetable.merge(
        rooms_df[["classroom_id", "building_name", "room_number", "room_type", "capacity"]],
        left_on="room_id",
        right_on="classroom_id",
        how="left"
    )
    st.dataframe(
        timetable[
            ["course_code", "exam_type", "num_students", "exam_day", "exam_time",
             "building_name", "room_number", "room_type", "capacity"]
        ]
    )

    # -----------------------------
    # Download Button
    # -----------------------------
    st.download_button(
        label="Download Example Timetable as CSV",
        data=timetable.to_csv(index=False),
        file_name='pareto_exam_timetable.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown(
    "**JIE42903 – Evolutionary Computing**  \n"
    "Multi-Objective Genetic Algorithm for University Exam Scheduling"
)
