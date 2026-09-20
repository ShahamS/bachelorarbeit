from .footprint import FootprintMatrix


class Node:
    def __init__(self, node_id, footprint_matrix=None, activities=None):
        self.node_id = node_id
        self.footprint_matrix = footprint_matrix or FootprintMatrix()
        self.activities = activities or set()

    def add_activity(self, activity):
        self.activities.add(activity)

    def add_relation(self, previous_activity, current_activity, register_activity=True):
        if register_activity:
            self.add_activity(current_activity)
        self.footprint_matrix.add_direct_succession(previous_activity, current_activity)

    def allows(self, previous_activity, current_activity):
        return self.footprint_matrix.allows(previous_activity, current_activity)
