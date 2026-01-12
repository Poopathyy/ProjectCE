def fitness_function(chromosome, exams, rooms):
    penalty = 0

    room_capacity = dict(zip(rooms['room_id'], rooms['capacity']))
    exam_students = dict(zip(exams['exam_id'], exams['num_students']))

    schedule = {}

    for exam, timeslot, room in chromosome:
        key = (timeslot, room)

        # Room capacity constraint
        if exam_students[exam] > room_capacity[room]:
            penalty += 1000

        # Time-slot conflict
        if key in schedule:
            penalty += 1000
        else:
            schedule[key] = exam

    return 1 / (1 + penalty)
