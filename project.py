import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ga import genetic_algorithm, create_population
from nsga2 import pareto_front

st.set_page_config(layout="wide")
st.title("University Exam Scheduling using Evolutionary Computation")

@st.cache_data
def load_data():
    return (
        pd.read_csv("exam_timeslot.csv"),
        pd.read_csv("classrooms.csv")
    )

exams, rooms = load_data()

st.sidebar.header("GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 50)
generations = st.sidebar.slider("Generations", 50, 500, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["Genetic Algorithm", "NSGA-II"]
)

if st.sidebar.button("Run"):
    if algorithm == "Genetic Algorithm":
        best, best_hist, avg_hist, runtime = genetic_algorithm(
            exams, rooms, population_size, generations, mutation_rate
        )

        st.subheader("GA Convergence")
        fig, ax = plt.subplots()
        ax.plot(best_hist, label="Best Fitness")
        ax.plot(avg_hist, label="Average Fitness")
        ax.legend()
        st.pyplot(fig)

        st.success(f"Runtime: {runtime:.2f} seconds")

        st.subheader("Best Exam Schedule")
        result = []
        for i, r in enumerate(best):
            result.append({
                "Course": exams.iloc[i]["course_code"],
                "Day": exams.iloc[i]["exam_day"],
                "Time": exams.iloc[i]["exam_time"],
                "Room": rooms.iloc[r]["room_number"]
            })
        st.dataframe(pd.DataFrame(result))

    else:
        population = create_population(
            population_size,
            len(exams),
            len(rooms)
        )
        front = pareto_front(population, exams, rooms)

        x = [f[1][0] for f in front]
        y = [f[1][1] for f in front]

        st.subheader("NSGA-II Pareto Front")
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel("Conflicts")
        ax.set_ylabel("Underutilization")
        st.pyplot(fig)
