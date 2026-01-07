from constraints import (
    room_capacity_violation,
    room_time_conflict,
    room_underutilization
)

def fitness(chromosome, exams, rooms):
    penalty = 0
    schedule = {}

    for exam_idx, room_idx in enumerate(chromosome):
        exam = exams.iloc[exam_idx]
        room = rooms.iloc[room_idx]

        key = (exam["exam_day"], exam["exam_time"], room_idx)

        # Hard constraint: same room & time
        if room_time_conflict(schedule, key):
            penalty += 200
        else:
            schedule[key] = exam_idx

        # Hard constraint: capacity
        penalty += 100 * room_capacity_violation(
            exam["num_students"],
            room["capacity"]
        )

        # Soft constraint: underutilization
        penalty += 10 * room_underutilization(
            exam["num_students"],
            room["capacity"]
        )

    return 1 / (1 + penalty)
