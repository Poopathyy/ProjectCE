import pandas as pd
import random

def load_data():
    exams = pd.read_csv("data/exam_timeslot.csv")
    rooms = pd.read_csv("data/classrooms.csv")
    return exams, rooms

def random_assignment(exams, rooms, timeslots):
    chromosome = []
    for exam in exams['exam_id']:
        timeslot = random.choice(timeslots)
        room = random.choice(rooms['room_id'].tolist())
        chromosome.append((exam, timeslot, room))
    return chromosome
