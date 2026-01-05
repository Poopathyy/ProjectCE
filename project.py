import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import sys
import io

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    st.error("⚠️ DEAP library is not installed. Please install it using: `pip install deap`")

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

# Check if DEAP is installed
if not DEAP_AVAILABLE:
    st.stop()

# =====================================
# Load CSV Files
# =====================================
@st.cache_data
def load_data():
    try:
        room_df = pd.read_csv("classrooms.csv")
        
        # Try to load exam data, create sample if not found
        try:
            exam_df = pd.read_csv("exam_timeslot.csv")
        except FileNotFoundError:
            exam_df = generate_sample_exam_data()
            st.info("📝 Using generated sample exam data. To use your own data, create 'exam_timeslot.csv'")
        
        return exam_df, room_df
    except FileNotFoundError:
        st.error("❌ 'classrooms.csv' not found. Please ensure it's in the same directory.")
        return None, None

def generate_sample_exam_data(num_students=200, num_courses=15):
    """Generate sample exam registration data"""
    courses = [f'CS{101 + i}' for i in range(num_courses)]
    students = [f'S{1000 + i}' for i in range(num_students)]
    
    data = []
    for course in courses:
        num_course_students = random.randint(15, 45)
        course_students = random.sample(students, min(num_course_students, len(students)))
        
        for student in course_students:
            if random.random() < 0.3:  # 30% chance to take this course
                data.append({
                    'student_id': student,
                    'course_code': course,
                    'timeslot': random.choice([1, 2, 3, 4, 5])
                })
    
    return pd.DataFrame(data)

exam_df, room_df = load_data()

if exam_df is None or room_df is None:
    st.stop()

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
# Add room_name if not exists
if 'room_name' not in room_df.columns:
    room_df['room_name'] = room_df['building_name'] + '-' + room_df['room_number'].astype(str)

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
    if hasattr(creator, attr):
        delattr(creator, attr)

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
# Visualization Functions (Matplotlib only)
# =====================================
def plot_convergence(log):
    """Plot convergence curve using Matplotlib"""
    gens = log.select("gen")
    min_fit = log.select("min")
    avg_fit = log.select("avg")
    max_fit = log.select("max")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(gens, min_fit, 'g-', linewidth=3, label='Best Fitness', marker='o', markersize=4)
    ax.plot(gens, avg_fit, 'b--', linewidth=2, label='Average Fitness')
    ax.plot(gens, max_fit, 'r:', linewidth=1, label='Worst Fitness')
    
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (Penalty Score)", fontsize=12)
    ax.set_title("GA Convergence Curve", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set background color
    fig.patch.set_facecolor('#f0f2f6')
    ax.set_facecolor('white')
    
    return fig

def plot_pareto_front(pareto_pop):
    """Plot Pareto front for NSGA-II results"""
    f1 = [ind.fitness.values[0] for ind in pareto_pop]
    f2 = [ind.fitness.values[1] for ind in pareto_pop]
    
    # Get non-dominated solutions
    pareto_front = tools.sortNondominated(pareto_pop, k=len(pareto_pop), first_front_only=True)[0]
    pf_f1 = [ind.fitness.values[0] for ind in pareto_front]
    pf_f2 = [ind.fitness.values[1] for ind in pareto_front]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all solutions
    ax.scatter(f1, f2, c='lightblue', s=50, alpha=0.6, edgecolors='black', linewidth=0.5, label='All Solutions')
    
    # Plot Pareto front
    # Sort for better line visualization
    pf_sorted = sorted(zip(pf_f1, pf_f2), key=lambda x: x[0])
    if pf_sorted:
        pf_x, pf_y = zip(*pf_sorted)
        ax.plot(pf_x, pf_y, 'r-', linewidth=2, label='Pareto Front')
        ax.scatter(pf_x, pf_y, c='red', s=100, edgecolors='black', linewidth=1, zorder=5)
    
    ax.set_xlabel("Student Clashes", fontsize=12)
    ax.set_ylabel("Capacity Violations", fontsize=12)
    ax.set_title("NSGA-II Pareto Front", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set background color
    fig.patch.set_facecolor('#f0f2f6')
    ax.set_facecolor('white')
    
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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    cmap = plt.cm.viridis
    im = ax.imshow(schedule_matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Number of Exams', rotation=-90, va="bottom")
    
    # Set labels
    ax.set_xlabel('Rooms', fontsize=12)
    ax.set_ylabel('Time Slots', fontsize=12)
    ax.set_title('Exam Schedule Heatmap', fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(len(ROOMS[:10])))
    ax.set_yticks(np.arange(len(TIMESLOTS)))
    ax.set_xticklabels(ROOMS[:10], rotation=45, ha='right')
    ax.set_yticklabels([f'Slot {ts}' for ts in TIMESLOTS])
    
    # Add text annotations
    for i in range(len(TIMESLOTS)):
        for j in range(len(ROOMS[:10])):
            if schedule_matrix[i][j] > 0:
                ax.text(j, i, schedule_matrix[i][j], 
                       ha="center", va="center", color="white", fontweight='bold')
    
    fig.tight_layout()
    return fig

def plot_room_capacity_distribution():
    """Plot room capacity distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(room_df['capacity'], bins=10, color='#1E88E5', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Room Capacity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Room Capacity Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    fig.patch.set_facecolor('#f0f2f6')
    ax.set_facecolor('white')
    
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
    
    fig_cap = plot_room_capacity_distribution()
    st.pyplot(fig_cap)

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
        st.subheader("📉 Convergence Curve")
        conv_fig = plot_convergence(log)
        st.pyplot(conv_fig)
        
        # Schedule visualization
        st.subheader("📅 Best Schedule Visualization")
        schedule_fig = visualize_schedule(best_individual)
        st.pyplot(schedule_fig)
        
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
                'Room Capacity': room_capacity.get(room, 0),
                'Capacity OK': '✅' if exam_students.get(exam, 0) <= room_capacity.get(room, 999) else '❌'
            })
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df, use_container_width=True)
        
        # Export schedule option
        csv = schedule_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Schedule as CSV",
            data=csv,
            file_name="optimized_schedule.csv",
            mime="text/csv",
        )

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
        st.pyplot(pareto_fig)
        
        # Show selected solutions from Pareto front
        st.subheader("🎯 Selected Pareto Solutions")
        
        if len(pareto_front) > 0:
            # Create a selection interface
            options = [f"Solution {i+1}: Clashes={pareto_front[i].fitness.values[0]}, Violations={pareto_front[i].fitness.values[1]}" 
                      for i in range(len(pareto_front))]
            
            selected_option = st.selectbox(
                "Select a solution from Pareto front:",
                options=options,
                index=0
            )
            
            selected_idx = options.index(selected_option)
            selected_solution = pareto_front[selected_idx]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Student Clashes", selected_solution.fitness.values[0])
            with col2:
                st.metric("Capacity Violations", selected_solution.fitness.values[1])
            
            # Visualize this schedule
            st.subheader("📅 Selected Schedule Visualization")
            selected_schedule_fig = visualize_schedule(selected_solution)
            st.pyplot(selected_schedule_fig)

# =====================================
# Requirements Section
# =====================================
with st.sidebar:
    st.markdown("---")
    st.subheader("📦 Installation")
    
    if st.button("Show Requirements"):
        st.code("""
pip install streamlit pandas numpy matplotlib deap
""", language="bash")
    
    st.markdown("---")
    st.caption("JIE42903 Evolutionary Computing")

# =====================================
# Footer
# =====================================
st.markdown("---")
st.markdown(
    """
    **JIE42903 Evolutionary Computing Project**  
    *University Exam Scheduling using Genetic Algorithms*  
    Developed with ❤️ using Streamlit, DEAP, and Matplotlib
    """
)
