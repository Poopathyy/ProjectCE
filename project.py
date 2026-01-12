import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ga_scheduler import run_ga

st.title("University Exam Scheduling using Genetic Algorithm")

exams = pd.read_csv("exam_timeslot.csv")
rooms = pd.read_csv("classrooms.csv")

st.write("Exam Dataset Columns:", exams.columns.tolist())
st.write("Room Dataset Columns:", rooms.columns.tolist())

generations = st.slider("Generations", 50, 500, 200)
population = st.slider("Population Size", 20, 100, 50)

if st.button("Run GA"):
    best, history = run_ga(exams, rooms, generations, population)

    st.subheader("Fitness Convergence")
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    st.pyplot(plt)

    timetable = pd.DataFrame(best)
    st.subheader("Optimized Timetable")
    st.dataframe(timetable)
