import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# -----------------------------
# Load datasets
# -----------------------------
@st.cache_data
def load_data():
    rooms = pd.read_csv("classrooms.csv")   # your classroom dataset
    exams = pd.read_csv("exam_timeslot.csv")
    return rooms, exams

rooms_df, exams_df = load_data()

NUM_EXAMS = len(exams_df)
ROOM_IDS = rooms_df["classroom_id"].tolist()

# -----------------------------
# Fitness Function
# -----------------------------
def fitness_function(chromosome):
    penalty = 0

    # Exam-room penalties
    for i, exam in exams_df.iterrows():
        room = rooms_df[rooms_df["classroom_id"] == chromosome[i]].iloc[0]

        # Capacity violation (hard)
        if exam["num_students"] > room["capacity"]:
            penalty += (exam["num_students"] - room["capacity"]) * 10

        # Room type mismatch (soft)
        if exam["exam_type"] == "Practical" and room["room_type"] != "Lab":
            penalty += 5

        # High utilization penalty
        if exam["num_students"] > 0.8 * room["capacity"]:
            penalty += 3

    # Room-time clash penalty
    for i in range(NUM_EXAMS):
        for j in range(i + 1, NUM_EXAMS):
            if (
                chromosome[i] == chromosome[j]
                and exams_df.iloc[i]["exam_day"] == exams_df.iloc[j]["exam_day"]
                and exams_df.iloc[i]["exam_time"] == exams_df.iloc[j]["exam_time"]
            ):
                penalty += 50

    return penalty

# -----------------------------
# Genetic Algorithm Components
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

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]

# -----------------------------
# GA Execution
# -----------------------------
def run_ga(pop_size, generations, mutation_rate):
    population = [create_individual() for _ in range(pop_size)]
    best_fitness_history = []

    for gen in range(generations):
        fitnesses = [fitness_function(ind) for ind in population]
        best_fitness_history.append(min(fitnesses))

        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    best_index = fitnesses.index(min(fitnesses))
    return population[best_index], best_fitness_history

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📅 University Exam Scheduling using Genetic Algorithm")

st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 80)
generations = st.sidebar.slider("Generations", 50, 500, 200)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

if st.sidebar.button("Run Genetic Algorithm"):
    with st.spinner("Optimizing exam schedule..."):
        best_solution, fitness_history = run_ga(
            population_size, generations, mutation_rate
        )

    st.success("Optimization Completed!")

    # -----------------------------
    # Convergence Plot
    # -----------------------------
    st.subheader("📉 Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(fitness_history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Penalty)")
    st.pyplot(fig)

    # -----------------------------
    # Final Timetable
    # -----------------------------
    st.subheader("🗓️ Final Exam Timetable")

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
            [
                "course_code",
                "exam_type",
                "num_students",
                "exam_day",
                "exam_time",
                "building_name",
                "room_number",
                "room_type",
                "capacity",
            ]
        ]
    )

    st.subheader("✅ Best Fitness Score")
    st.metric(label="Total Penalty", value=fitness_history[-1])

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "**JIE42903 – Evolutionary Computing**  \n"
    "Genetic Algorithm based University Exam Scheduling"
)
