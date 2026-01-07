def evaluate_objectives(chromosome, exams, rooms):
    conflicts = 0
    underutilization = 0
    schedule = {}

    for exam_idx, room_idx in enumerate(chromosome):
        exam = exams.iloc[exam_idx]
        room = rooms.iloc[room_idx]

        key = (exam["exam_day"], exam["exam_time"], room_idx)

        if key in schedule:
            conflicts += 1
        schedule[key] = exam_idx

        underutilization += 1 - (exam["num_students"] / room["capacity"])

    return conflicts, underutilization


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_front(population, exams, rooms):
    scored = [(c, evaluate_objectives(c, exams, rooms)) for c in population]
    front = []

    for i, (c1, s1) in enumerate(scored):
        dominated = False
        for j, (c2, s2) in enumerate(scored):
            if i != j and dominates(s2, s1):
                dominated = True
                break
        if not dominated:
            front.append((c1, s1))

    return front
