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
# Fitness Function with Detailed Penalty Breakdown
# -----------------------------
def fitness_function(chromosome):
    penalties = {
        "capacity_violation": 0,
        "room_type_mismatch": 0,
        "high_utilization": 0,
        "room_time_clash": 0
    }

    # Exam-room penalties
    for i, exam in exams_df.iterrows():
        room = rooms_df[rooms_df["classroom_id"] == chromosome[i]].iloc[0]

        # Capacity violation
        if exam["num_students"] > room["capacity"]:
            penalties["capacity_violation"] += (exam["num_students"] - room["capacity"]) * 10

        # Room type mismatch
        if exam["exam_type"] == "Practical" and room["room_type"] != "Lab":
            penalties["room_type_mismatch"] += 5

        # High utilization
        if exam["num_students"] > 0.8 * room["capacity"]:
            penalties["high_utilization"] += 3

    # Room-time clash penalty
    for i in range(NUM_EXAMS):
        for j in range(i + 1, NUM_EXAMS):
            if (
                chromosome[i] == chromosome[j]
                and exams_df.iloc[i]["exam_day"] == exams_df.iloc[j]["exam_day"]
                and exams_df.iloc[i]["exam_time"] == exams_df.iloc[j]["exam_time"]
            ):
                penalties["room_time_clash"] += 50

    total_penalty = sum(penalties.values())
    return total_penalty, penalties

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

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]

# -----------------------------
# GA Execution
# -----------------------------
def run_ga(pop_size, generations, mutation_rate):
    population = [create_individual() for _ in range(pop_size)]
    best_fitness_history = []
    best_individual_overall = None
    best_penalties_overall = None
    best_fitness_overall = float('inf')

    for gen in range(generations):
        fitnesses = []
        penalties_list = []
        for ind in population:
            total, penalties = fitness_function(ind)
            fitnesses.append(total)
            penalties_list.append(penalties)

        min_fitness = min(fitnesses)
        best_fitness_history.append(min_fitness)

        # Track overall best
        if min_fitness < best_fitness_overall:
            best_fitness_overall = min_fitness
            best_individual_overall = population[fitnesses.index(min_fitness)]
            best_penalties_overall = penalties_list[fitnesses.index(min_fitness)]

        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    return best_individual_overall, best_fitness_history, best_penalties_overall

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Exam Scheduling GA", layout="wide")
st.title("📅 University Exam Scheduling using Genetic Algorithm")

# Sidebar - GA parameters
st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 80)
generations = st.sidebar.slider("Generations", 50, 500, 200)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

if st.sidebar.button("Run Genetic Algorithm"):
    with st.spinner("Optimizing exam schedule..."):
        best_solution, fitness_history, penalty_breakdown = run_ga(
            population_size, generations, mutation_rate
        )

    st.success("Optimization Completed!")

    # -----------------------------
    # Fitness Convergence Plot
    # -----------------------------
    st.subheader("📉 Fitness Convergence")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fitness_history, color='blue', marker='o', markersize=3)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Total Penalty")
    ax.set_title("GA Fitness Convergence")
    st.pyplot(fig)

    # -----------------------------
    # Final Exam Timetable
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
                "course_code", "exam_type", "num_students",
                "exam_day", "exam_time", "building_name",
                "room_number", "room_type", "capacity"
            ]
        ]
    )

    # -----------------------------
    # Total Penalty & Breakdown
    # -----------------------------
    st.subheader("✅ Best Fitness Score with Detailed Penalties")
    st.metric(label="Total Penalty", value=sum(penalty_breakdown.values()))

    st.markdown("**Breakdown by Constraint:**")
    st.write(penalty_breakdown)

    # -----------------------------
    # Download Button
    # -----------------------------
    st.download_button(
        label="Download Timetable as CSV",
        data=timetable.to_csv(index=False),
        file_name='exam_timetable.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown(
    "**JIE42903 – Evolutionary Computing**  \n"
    "Genetic Algorithm based University Exam Scheduling"
)
