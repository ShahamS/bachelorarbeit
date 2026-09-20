from collections import defaultdict


START_ACTIVITY = "__START__"
END_ACTIVITY = "__END__"


class FootprintMatrix:
    def __init__(self):
        self.allowed_predecessors = defaultdict(set)

    def add_direct_succession(self, previous_activity, current_activity):
        self.allowed_predecessors[current_activity].add(previous_activity)

    def allows(self, previous_activity, current_activity):
        return previous_activity in self.allowed_predecessors.get(current_activity, set())

    def activities(self):
        return set(self.allowed_predecessors.keys())

    def to_dict(self):
        return {
            current_activity: set(predecessors)
            for current_activity, predecessors in self.allowed_predecessors.items()
        }
