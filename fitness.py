def fitness_function(chromosome, rooms):
    hard_penalty = 0
    soft_penalty = 0

    room_capacity = dict(zip(rooms["room_id"], rooms["capacity"]))
    schedule = {}

    for gene in chromosome:
        key = (gene["timeslot"], gene["room"])

        # HARD: room capacity
        if gene["students"] > room_capacity[gene["room"]]:
            hard_penalty += 1

        # HARD: same room, same timeslot
        if key in schedule:
            hard_penalty += 1
        else:
            schedule[key] = gene["exam"]

        # SOFT: room underutilization
        unused = room_capacity[gene["room"]] - gene["students"]
        soft_penalty += unused / room_capacity[gene["room"]]

    # MULTI-OBJECTIVE (weighted)
    fitness = 1 / (1 + 1000 * hard_penalty + 10 * soft_penalty)
    return fitness

