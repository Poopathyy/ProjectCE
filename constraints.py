def room_capacity_violation(num_students, capacity):
    if num_students > capacity:
        return num_students - capacity
    return 0


def room_time_conflict(schedule, key):
    return key in schedule


def room_underutilization(num_students, capacity):
    return 1 - (num_students / capacity)
