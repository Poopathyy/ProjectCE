import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# ===============================
# Load Data
# ===============================
exams = pd.read_csv("exam_timeslot.csv")
rooms = pd.read_csv("classrooms.csv")

exams["timeslot"] = exams["exam_day"].astype(str) + "_" + exams["exam_time"].astype(str)
TIMESLOTS = exams["timeslot"].unique().tolist()

NUM_EXAMS = len(exams)
NUM_ROOMS = len(rooms)
NUM_TIMESLOTS = len(TIMESLOTS)

# ===============================
# Genetic Algorithm Components
# ===============================
def create_individual():
    return [(random.randint(0, NUM_TIMESLOTS - 1),
             random.randint(0, NUM_ROOMS - 1)) for _ in range(NUM_EXAMS)]

def create_population(size):
    return [create_individual() for _ in range(size)]

def fitness_components(individual):
    cap_violation = 0
    conflict = 0
    type_penalty = 0
    unused_capacity = 0
    used = {}

    for i, (slot, room) in enumerate(individual):
        exam = exams.iloc[i]
        classroom = rooms.iloc[room]

        if (slot, room) in used:
            conflict += 1
        used[(slot, room)] = 1

        if exam["num_students"] > classroom["capacity"]:
            cap_violation += exam["num_students"] - classroom["capacity"]

        if exam["exam_type"] != classroom["room_type"]:
            type_penalty += 1

        unused_capacity += max(0, classroom["capacity"] - exam["num_students"])

    return cap_violation, conflict, type_penalty, unused_capacity

def fitness(ind):
    cap, con, type_p, unused = fitness_components(ind)
    penalty = 1000*cap + 1000*con + 10*type_p + unused
    return 1 / (1 + penalty)

def multi_objective_fitness(ind):
    cap, con, type_p, unused = fitness_components(ind)
    return cap + con, type_p + unused

def tournament_selection(pop):
    return max(random.sample(pop, 3), key=fitness)

def crossover(p1, p2):
    point = random.randint(1, NUM_EXAMS - 1)
    return p1[:point] + p2[point:]

def mutate(ind, rate):
    for i in range(NUM_EXAMS):
        if random.random() < rate:
            ind[i] = (random.randint(0, NUM_TIMESLOTS - 1),
                      random.randint(0, NUM_ROOMS - 1))
    return ind

def genetic_algorithm(pop_size, generations, mutation_rate):
    population = create_population(pop_size)
    best_history = []

    for _ in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = mutate(crossover(p1, p2), mutation_rate)
            new_pop.append(child)
        population = new_pop
        best_history.append(fitness(max(population, key=fitness)))

    return max(population, key=fitness), best_history, population

# ===============================
# Pareto Front
# ===============================
def dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def pareto_front(pop):
    scores = [multi_objective_fitness(ind) for ind in pop]
    pareto = []
    for i, s in enumerate(scores):
        if not any(dominates(scores[j], s) for j in range(len(scores)) if j != i):
            pareto.append(s)
    return pareto

# ===============================
# Streamlit UI
# ===============================
st.title("📘 University Exam Scheduling – GA Dashboard")

st.markdown("""
This interactive dashboard visualizes the **performance and trade-offs** of a  
**Genetic Algorithm (GA)** applied to the **University Exam Scheduling problem**.
""")

# Sidebar Controls
st.sidebar.header("GA Parameter Controls")
pop_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Number of Generations", 50, 300, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

if st.button("🚀 Run Optimization"):
    best, history, final_population = genetic_algorithm(
        pop_size, generations, mutation_rate
    )

     # ---- Best Fitness Score ----
    best_fitness_score = fitness(best)
    st.subheader("🏆 Best Fitness Score")
    st.metric("Best Fitness Achieved", f"{best_fitness_score:.6f}")

    # -------- Convergence Curve --------
    st.subheader("📈 Convergence Curve")
    plt.figure()
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    st.pyplot(plt)

    # -------- Pareto Front --------
    st.subheader("⚖️ Pareto Front – Trade-off Visualization")
    pareto = pareto_front(final_population)

    x = [p[0] for p in pareto]
    y = [p[1] for p in pareto]

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Hard Constraint Violations")
    plt.ylabel("Scheduling Quality Penalty")
    plt.title("Pareto Front of Exam Schedules")
    st.pyplot(plt)

    # -------- Best Solution Details --------
    st.subheader("📋 Best Exam Timetable (with Venue)")

    schedule = []
    
    for i, (slot, room) in enumerate(best):
        schedule.append({
            "Exam ID": exams.iloc[i]["exam_id"],
            "Course": exams.iloc[i]["course_code"],
            "Day & Time": TIMESLOTS[slot],
            "Venue": f"{rooms.iloc[room]['building_name']} - Room {rooms.iloc[room]['room_number']}",
            "Room Type": rooms.iloc[room]["room_type"],
            "Room Capacity": rooms.iloc[room]["capacity"],
            "Students": exams.iloc[i]["num_students"]
        })
    
    st.dataframe(pd.DataFrame(schedule))

    # -------- Constraint Summary --------
    st.subheader("⚠️ Solution Quality Summary")
    cap, con, type_p, unused = fitness_components(best)

    st.write(f"Hard Constraint Violations: **{cap + con}**")
    st.write(f"Room Type Mismatch Penalty: **{type_p}**")
    st.write(f"Unused Room Capacity: **{unused}**")

