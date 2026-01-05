import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import time

from deap import base, creator, tools, algorithms

# =====================================
# Streamlit Page Config
# =====================================
st.set_page_config(
    page_title="University Exam Scheduling (GA)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 University Exam Scheduling using Evolutionary Algorithms")
st.markdown("**JIE42903 - Evolutionary Computing Project**")

# =====================================
# Load CSV Files
# =====================================
@st.cache_data
def load_data():
    try:
        exam_df = pd.read_csv("exam_timeslot.csv")
        room_df = pd.read_csv("classrooms.csv")
        
        # If exam_timeslot.csv doesn't exist, create sample data
        if exam_df.empty:
            exam_df = generate_sample_exam_data()
        
        return exam_df, room_df
    except FileNotFoundError:
        st.error("Required CSV files not found. Using sample data.")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data if CSV files are missing"""
    # Sample classrooms (from your provided data)
    room_df = pd.DataFrame({
        'classroom_id': [1, 2, 3, 4, 5],
        'building_name': ['A', 'K', 'B', 'A', 'I'],
        'room_number': [305, 144, 710, 541, 747],
        'capacity': [35, 24, 46, 30, 35],
        'room_type': ['Lecture Hall', 'Classroom', 'Classroom', 'Classroom', 'Lecture Hall']
    })
    room_df['room_name'] = room_df['building_name'] + '-' + room_df['room_number'].astype(str)
    
    # Generate sample exam data
    exam_df = generate_sample_exam_data()
    
    return exam_df, room_df

def generate_sample_exam_data(num_students=200, num_courses=15):
    """Generate sample exam registration data"""
    courses = [f'CS{101 + i}' for i in range(num_courses)]
    students = [f'S{1000 + i}' for i in range(num_students)]
    
    data = []
    for course in courses:
        # Each course has 15-45 students
        num_course_students = random.randint(15, 45)
        course_students = random.sample(students, min(num_course_students, len(students)))
        
        for student in course_students:
            # Students typically have 3-6 exams
            if random.random() < 0.3:  # 30% chance to take this course
                data.append({
                    'student_id': student,
                    'course_code': course,
                    'timeslot': random.choice([1, 2, 3, 4, 5])  # For initial preference
                })
    
    return pd.DataFrame(data)

exam_df, room_df = load_data()

# =====================================
# Sidebar: Data Overview & Parameters
# =====================================
st.sidebar.header("📊 Data Overview")

# Create tabs in sidebar
tab1, tab2 = st.sidebar.tabs(["📈 Statistics", "⚙ Parameters"])

with tab1:
    st.metric("Number of Exams", exam_df["course_code"].nunique())
    st.metric("Number of Students", exam_df["student_id"].nunique())
    st.metric("Available Rooms", room_df.shape[0])
    
    # Show room capacity distribution
    room_cap_stats = room_df['capacity'].describe()
    st.write("**Room Capacity Stats:**")
    st.write(f"Min: {int(room_cap_stats['min'])}")
    st.write(f"Max: {int(room_cap_stats['max'])}")
    st.write(f"Avg: {room_cap_stats['mean']:.1f}")

with tab2:
    st.subheader("⚙ Algorithm Parameters")
    
    POP_SIZE = st.slider("Population Size", 50, 500, 100, 50)
    NGEN = st.slider("Generations", 20, 500, 100, 20)
    CX_PB = st.slider("Crossover Probability", 0.1, 1.0, 0.8, 0.05)
    MUT_PB = st.slider("Mutation Probability", 0.01, 0.5, 0.2, 0.01)
    TOURNSIZE = st.slider("Tournament Size", 2, 10, 3)
    
    st.subheader("🎯 Objective Weights")
    clash_weight = st.slider("Student Clash Weight", 1, 20, 10, 1)
    capacity_weight = st.slider("Capacity Violation Weight", 1, 20, 5, 1)
    room_conflict_weight = st.slider("Room Conflict Weight", 1, 20, 5, 1)

# =====================================
# Preprocessing
# =====================================
EXAMS = sorted(exam_df["course_code"].unique())
TIMESLOTS = list(range(1, 11))  # 10 time slots
ROOMS = room_df["room_name"].tolist()

room_capacity = dict(zip(room_df["room_name"], room_df["capacity"]))
exam_students = exam_df.groupby("course_code")["student_id"].nunique().to_dict()

# Create mapping for students and their exams
student_exams = defaultdict(list)
for _, row in exam_df.iterrows():
    student_exams[row['student_id']].append(row['course_code'])

# =====================================
# DEAP Setup
# =====================================
# Clear existing creators to avoid duplication errors
for attr in ['FitnessMin', 'Individual', 'FitnessMulti', 'IndividualMulti']:
    if attr in creator.__dict__:
        del creator.__dict__[attr]

# Single objective (minimization)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Multi-objective (minimize both objectives)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

# Attribute generators
toolbox.register("timeslot", random.choice, TIMESLOTS)
toolbox.register("room", random.choice, ROOMS)

# Individual and population
def create_individual():
    """Create an individual: list of (timeslot, room) pairs for each exam"""
    return [(random.choice(TIMESLOTS), random.choice(ROOMS)) for _ in range(len(EXAMS))]

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# For multi-objective
toolbox.register("individual_multi", tools.initIterate, creator.IndividualMulti, create_individual)
toolbox.register("population_multi", tools.initRepeat, list, toolbox.individual_multi)

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
        if len(exams_in_ts) > 1:
            # Get all students taking exams in this timeslot
            students_in_ts = exam_df[exam_df["course_code"].isin(exams_in_ts)]
            clash_penalty += students_in_ts.duplicated(subset=['student_id']).sum()
    
    # Room conflicts and capacity violations
    room_usage = defaultdict(list)  # (timeslot, room) -> list of exams
    
    for exam, (ts, room) in schedule.items():
        key = (ts, room)
        room_usage[key].append(exam)
        
        # Capacity check
        if exam_students[exam] > room_capacity[room]:
            capacity_penalty += 1
    
    # Room conflict penalty (multiple exams in same room at same time)
    for exams in room_usage.values():
        if len(exams) > 1:
            room_conflict_penalty += len(exams) - 1
    
    total_penalty = (
        clash_weight * clash_penalty +
        capacity_weight * capacity_penalty +
        room_conflict_weight * room_conflict_penalty
    )
    
    return (total_penalty,)

def multi_objective_fitness(individual):
    clashes = 0
    capacity_violations = 0
    
    schedule = {EXAMS[i]: individual[i] for i in range(len(EXAMS))}
    
    # Student clashes
    for ts in TIMESLOTS:
        exams_in_ts = [e for e, (t, _) in schedule.items() if t == ts]
        if len(exams_in_ts) > 1:
            students_in_ts = exam_df[exam_df["course_code"].isin(exams_in_ts)]
            clashes += students_in_ts.duplicated(subset=['student_id']).sum()
    
    # Capacity violations
    for exam, (_, room) in schedule.items():
        if exam_students[exam] > room_capacity[room]:
            capacity_violations += 1
    
    return clashes, capacity_violations

# Register operators
toolbox.register("evaluate", single_objective_fitness)
toolbox.register("evaluate_multi", multi_objective_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(TIMESLOTS)-1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

# =====================================
# Algorithm Execution Functions
# =====================================
def run_ga():
    """Run single-objective Genetic Algorithm"""
    with st.spinner("Running Genetic Algorithm..."):
        start_time = time.time()
        
        pop = toolbox.population(n=POP_SIZE)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
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
        
        execution_time = time.time() - start_time
        best_individual = hof[0]
        
        return log, best_individual, execution_time

def run_nsga2():
    """Run NSGA-II multi-objective optimization"""
    with st.spinner("Running NSGA-II..."):
        start_time = time.time()
        
        # Use multi-objective creators
        pop = toolbox.population_multi(n=POP_SIZE)
        
        # Evaluate initial population
        fitnesses = [toolbox.evaluate_multi(ind) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # NSGA-II algorithm
        for gen in range(NGEN):
            offspring = algorithms.varAnd(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB)
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [toolbox.evaluate_multi(ind) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            pop = tools.selNSGA2(pop + offspring, k=POP_SIZE)
        
        execution_time = time.time() - start_time
        return pop, execution_time

# =====================================
# Visualization Functions
# =====================================
def plot_convergence(log):
    """Plot convergence curve using Plotly"""
    gens = log.select("gen")
    min_fit = log.select("min")
    avg_fit = log.select("avg")
    max_fit = log.select("max")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=gens, y=min_fit,
        mode='lines+markers',
        name='Best Fitness',
        line=dict(color='green', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=gens, y=avg_fit,
        mode='lines',
        name='Average Fitness',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=gens, y=max_fit,
        mode='lines',
        name='Worst Fitness',
        line=dict(color='red', width=1, dash='dot')
    ))
    
    fig.update_layout(
        title="GA Convergence Curve",
        xaxis_title="Generation",
        yaxis_title="Fitness (Penalty Score)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def plot_pareto_front(pareto_pop):
    """Plot Pareto front for NSGA-II results"""
    f1 = [ind.fitness.values[0] for ind in pareto_pop]
    f2 = [ind.fitness.values[1] for ind in pareto_pop]
    
    # Get non-dominated solutions
    pareto_front = tools.sortNondominated(pareto_pop, k=len(pareto_pop), first_front_only=True)[0]
    pf_f1 = [ind.fitness.values[0] for ind in pareto_front]
    pf_f2 = [ind.fitness.values[1] for ind in pareto_front]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=f1, y=f2,
        mode='markers',
        name='All Solutions',
        marker=dict(color='lightblue', size=8, opacity=0.6)
    ))
    
    fig.add_trace(go.Scatter(
        x=pf_f1, y=pf_f2,
        mode='markers+lines',
        name='Pareto Front',
        marker=dict(color='red', size=12),
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="NSGA-II Pareto Front",
        xaxis_title="Student Clashes",
        yaxis_title="Capacity Violations",
        template="plotly_white",
        hovermode="closest"
    )
    
    return fig, pareto_front

def visualize_schedule(best_individual):
    """Create a schedule visualization"""
    schedule = {EXAMS[i]: best_individual[i] for i in range(len(EXAMS))}
    
    # Create schedule matrix
    schedule_matrix = []
    for ts in TIMESLOTS:
        row = []
        for room in ROOMS[:10]:  # Show first 10 rooms for clarity
            exams_in_slot = [e for e, (t, r) in schedule.items() if t == ts and r == room]
            row.append(len(exams_in_slot))
        schedule_matrix.append(row)
    
    fig = px.imshow(
        schedule_matrix,
        labels=dict(x="Rooms", y="Time Slots", color="Exams"),
        x=ROOMS[:10],
        y=[f"Slot {ts}" for ts in TIMESLOTS],
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    
    fig.update_layout(
        title="Exam Schedule Heatmap",
        xaxis_title="Rooms",
        yaxis_title="Time Slots"
    )
    
    return fig

# =====================================
# Main Streamlit Interface
# =====================================

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["🏫 Data Preview", "🎯 Single Objective GA", "📊 Multi-Objective NSGA-II"])

with tab1:
    st.subheader("📚 Data Preview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Classrooms Data**")
        st.dataframe(room_df.head(10), use_container_width=True)
        
    with col2:
        st.write("**Exam Registration Data**")
        st.dataframe(exam_df.head(10), use_container_width=True)
    
    # Data statistics
    st.subheader("📊 Data Statistics")
    
    fig1 = px.histogram(room_df, x='capacity', title='Room Capacity Distribution',
                       nbins=10, color_discrete_sequence=['#1E88E5'])
    fig1.update_layout(bargap=0.1)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("🎯 Single-Objective Genetic Algorithm")
    
    if st.button("▶ Run Genetic Algorithm", type="primary", key="run_ga"):
        log, best_individual, exec_time = run_ga()
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Fitness", f"{best_individual.fitness.values[0]:.2f}")
        with col2:
            st.metric("Execution Time", f"{exec_time:.2f} seconds")
        with col3:
            st.metric("Generations", NGEN)
        
        # Convergence plot
        st.plotly_chart(plot_convergence(log), use_container_width=True)
        
        # Schedule visualization
        st.subheader("📅 Best Schedule Visualization")
        st.plotly_chart(visualize_schedule(best_individual), use_container_width=True)
        
        # Detailed schedule table
        st.subheader("📋 Detailed Schedule")
        schedule_data = []
        for i, exam in enumerate(EXAMS):
            ts, room = best_individual[i]
            schedule_data.append({
                'Exam': exam,
                'Students': exam_students.get(exam, 0),
                'Time Slot': ts,
                'Room': room,
                'Room Capacity': room_capacity.get(room, 0)
            })
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df, use_container_width=True)

with tab3:
    st.subheader("📊 Multi-Objective Optimization with NSGA-II")
    
    if st.button("▶ Run NSGA-II", type="primary", key="run_nsga"):
        pareto_pop, exec_time = run_nsga2()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Execution Time", f"{exec_time:.2f} seconds")
        with col2:
            st.metric("Pareto Solutions", len(pareto_pop))
        
        # Pareto front visualization
        pareto_fig, pareto_front = plot_pareto_front(pareto_pop)
        st.plotly_chart(pareto_fig, use_container_width=True)
        
        # Show selected solutions from Pareto front
        st.subheader("🎯 Selected Pareto Solutions")
        
        if len(pareto_front) > 0:
            # Let user select which solution to view
            selected_idx = st.selectbox(
                "Select a solution from Pareto front:",
                range(len(pareto_front)),
                format_func=lambda i: f"Solution {i+1}: Clashes={pareto_front[i].fitness.values[0]}, Violations={pareto_front[i].fitness.values[1]}"
            )
            
            if selected_idx is not None:
                selected_solution = pareto_front[selected_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Student Clashes", selected_solution.fitness.values[0])
                with col2:
                    st.metric("Capacity Violations", selected_solution.fitness.values[1])
                
                # Visualize this schedule
                st.plotly_chart(visualize_schedule(selected_solution), use_container_width=True)

# =====================================
# Footer
# =====================================
st.markdown("---")
st.markdown(
    """
    **JIE42903 Evolutionary Computing Project**  
    *University Exam Scheduling using Genetic Algorithms*  
    Developed with ❤️ using Streamlit, DEAP, and Plotly
    """
)
