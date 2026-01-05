import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

# =====================================
# Streamlit Page Config
# =====================================
st.set_page_config(
    page_title="University Exam Scheduling (GA)",
    layout="wide"
)

st.title("🎓 University Exam Scheduling using Genetic Algorithm & NSGA-II")

# =====================================
# Load CSV Files
# =====================================
@st.cache_data
def load_data():
    exam_df = pd.read_csv("exam_timeslot.csv")
    room_df = pd.read_csv("classrooms.csv")
    return exam_df, room_df

exam_df, room_df = load_data()

st.sidebar.header("📂 Data Overview")
st.sidebar.write("Exams:", exam_df["course_code"].nunique())
st.sidebar.write("Students:", exam_df["student_id"].nunique())
st.sidebar.write("Rooms:", room_df["room_name"].nunique())

# =====================================
# Preprocessing
# =====================================
EXAMS = exam_df["course_code"].unique()
TIMESLOTS = exam_df["timeslot"].unique()
ROOMS = room_df["room_name"].unique()

room_capacity = dict(zip(room_df["room_name"], room_df["capacity"]))
exam_students = exam_df.groupby("course_code")["student_id"].nunique().to_dict()

# =====================================
# GA Parameters (Sidebar)
# =====================================
st.sidebar.header("⚙ GA Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 50, 200, 100)
NGEN = st.sidebar.slider("Generations", 20, 200, 100)
CX_PB = st.sidebar.slider("Crossover Probability", 0.1, 1.0, 0.8)
MUT_PB = st.sidebar.slider("Mutation Probability", 0.1, 1.0, 0.2)

# =====================================
# GA Setup
# =====================================
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("timeslot", random.choice, TIMESLOTS)
toolbox.register("room", random.choice, ROOMS)

toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    [(toolbox.timeslot, toolbox.room)],
    n=len(EXAMS)
)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =====================================
# Fitness Functions
# =====================================
def single_objective_fitness(individual):
    clash_penalty = 0
    capacity_penalty = 0
    room_conflict_penalty = 0

    schedule = {EXAMS[i]: individual[i] for i in range(len(EXAMS))}

    # Student clashes
    for ts in TIMESLOTS:
        exams_in_ts = [e for e, (t, _) in schedule.items() if t == ts]
        students = exam_df[exam_df["course_code"].isin(exams_in_ts)]["student_id"]
        clash_penalty += students.duplicated().sum()

    # Room capacity and conflicts
    used_rooms = {}
    for exam, (ts, room) in schedule.items():
        if exam_students[exam] > room_capacity[room]:
            capacity_penalty += 1

        key = (ts, room)
        if key in used_rooms:
            room_conflict_penalty += 1
        else:
            used_rooms[key] = exam

    total_penalty = (
        10 * clash_penalty +
        5 * capacity_penalty +
        5 * room_conflict_penalty
    )

    return (total_penalty,)

def multi_objective_fitness(individual):
    clashes = 0
    capacity_violations = 0

    schedule = {EXAMS[i]: individual[i] for i in range(len(EXAMS))}

    for ts in TIMESLOTS:
        exams_in_ts = [e for e, (t, _) in schedule.items() if t == ts]
        students = exam_df[exam_df["course_code"].isin(exams_in_ts)]["student_id"]
        clashes += students.duplicated().sum()

    for exam, (_, room) in schedule.items():
        if exam_students[exam] > room_capacity[room]:
            capacity_violations += 1

    return clashes, capacity_violations

toolbox.register("evaluate", single_objective_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# =====================================
# Run Single Objective GA
# =====================================
def run_ga():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CX_PB,
        mutpb=MUT_PB,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=False
    )
    return log, hof[0]

# =====================================
# Run NSGA-II
# =====================================
def run_nsga():
    pop = toolbox.population(n=POP_SIZE)

    for ind in pop:
        ind.fitness.values = multi_objective_fitness(ind)

    for _ in range(NGEN):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB)
        for ind in offspring:
            ind.fitness.values = multi_objective_fitness(ind)
        pop = tools.selNSGA2(pop + offspring, k=len(pop))

    return pop

# =====================================
# Streamlit Actions
# =====================================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Run Genetic Algorithm"):
        log, best = run_ga()

        gens = log.select("gen")
        min_fit = log.select("min")
        avg_fit = log.select("avg")

        st.subheader("📉 GA Convergence Curve")
        fig, ax = plt.subplots()
        ax.plot(gens, min_fit, label="Minimum Fitness")
        ax.plot(gens, avg_fit, label="Average Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Penalty Score")
        ax.legend()
        st.pyplot(fig)

        st.success("Best Fitness Score: " + str(best.fitness.values[0]))

with col2:
    if st.button("▶ Run NSGA-II"):
        pareto_pop = run_nsga()

        f1 = [ind.fitness.values[0] for ind in pareto_pop]
        f2 = [ind.fitness.values[1] for ind in pareto_pop]

        st.subheader("📊 NSGA-II Pareto Front")
        fig, ax = plt.subplots()
        ax.scatter(f1, f2)
        ax.set_xlabel("Student Clashes")
        ax.set_ylabel("Capacity Violations")
        st.pyplot(fig)

# =====================================
# Footer
# =====================================
st.markdown("---")
st.markdown(
    "**Evolutionary Computing Project – University Exam Scheduling**  \n"
    "Genetic Algorithm & NSGA-II with Streamlit Visualization"
)
