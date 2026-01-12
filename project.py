import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ga_scheduler import run_ga

st.title("University Exam Scheduling using Genetic Algorithm")

exams = pd.read_csv("exam_timeslot.csv")
rooms = pd.read_csv("classrooms.csv")

timeslots = exams['timeslot'].unique().tolist()

generations = st.slider("Generations", 50, 500, 200)
population = st.slider("Population Size", 20, 100, 50)

if st.button("Run Genetic Algorithm"):
    best_solution, history = run_ga(
        exams, rooms, timeslots,
        generations, population
    )

    st.subheader("GA Convergence")
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    st.pyplot(plt)

    df = pd.DataFrame(
        best_solution,
        columns=["Exam", "Time Slot", "Room"]
    )
    st.subheader("Optimized Exam Timetable")
    st.dataframe(df)

    df.to_csv("sample_timetable.csv", index=False)
