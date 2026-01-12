import random

def generate_timeslots(num_slots=10):
    return list(range(1, num_slots + 1))

def random_assignment(exams, rooms, timeslots):
    chromosome = []
    for _, row in exams.iterrows():
        chromosome.append({
            "exam": row["exam_id"],
            "students": row["num_students"],
            "timeslot": random.choice(timeslots),
            "room": random.choice(rooms["room_id"].tolist())
        })
    return chromosome
