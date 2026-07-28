from .footprint import FootprintMatrix


class Node:
    # Ein lokaler Knoten im verteilten CheckMyFlow-Modell

    def __init__(self, node_id, footprint_matrix=None, activities=None):
        self.node_id = node_id
        self.footprint_matrix = footprint_matrix or FootprintMatrix()
        self.activities = activities or set()

    def add_activity(self, activity):
        # Merkt, dass diese Aktivität zu diesem Knoten gehört

        self.activities.add(activity)

    def add_relation(self, previous_activity, current_activity):
        # Speichert eine erlaubte direkte Nachfolge in der lokalen Matrix

        self.add_activity(current_activity)
        self.footprint_matrix.add_direct_succession(previous_activity, current_activity)

    def allows(self, previous_activity, current_activity):
        # Delegiert die Konformitätsfrage an die lokale Footprint-Matrix

        return self.footprint_matrix.allows(previous_activity, current_activity)
