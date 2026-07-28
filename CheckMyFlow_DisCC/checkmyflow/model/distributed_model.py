from .footprint import START_ACTIVITY
from .node import Node


class DistributedFootprintModel:
    # Verwaltet die lokalen Footprint-Matrizen aller Nodes
    def __init__(self):
        self.nodes = {}
        self.activity_to_node = {}

    def get_or_create_node(self, node_id):
        # Liefert eine vorhandene Node oder legt sie neu an
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id=node_id)
        return self.nodes[node_id]

    def register_activity(self, activity, node_id):
        # Speichert, welche Node für eine Aktivität zuständig ist
        node = self.get_or_create_node(node_id)
        node.add_activity(activity)
        self.activity_to_node[activity] = node_id

    def add_event_relation(self, current_event, previous_event=None):
        # Lernt eine Relation für ein aktuelles Event.
        # Für das erste Event eines Traces wird ein künstlicher `START_ACTIVITY`-Vorgänger verwendet
        current_node_id = current_event.node_id
        previous_activity = (
            START_ACTIVITY if previous_event is None else previous_event.activity
        )
        # Registriere die aktuelle Aktivität bei der Node des aktuellen Events
        self.register_activity(current_event.activity, current_node_id)
        # Füge die Relation zwischen dem vorherigen Event und dem aktuellen Event bei der Node des aktuellen Events hinzu
        node = self.get_or_create_node(current_node_id)
        node.add_relation(previous_activity, current_event.activity)

    def get_node_for_event(self, event):
        # Findet die Node, die für ein Event zuständig ist

        return self.get_or_create_node(event.node_id)

    def get_existing_node_for_event(self, event):
        # Findet die bekannte Node eines Events, ohne das Modell zu verändern

        return self.nodes.get(event.node_id)

    def get_node_for_activity(self, activity):
        # Findet die bekannte Node für eine Aktivität.

        node_id = self.activity_to_node.get(activity)
        if node_id is None:
            return None
        return self.nodes.get(node_id)

    def allows(self, previous_activity, current_activity):
        # Prüft eine Relation über die Node der aktuellen Aktivität.

        node = self.get_node_for_activity(current_activity)
        if node is None:
            return False
        return node.allows(previous_activity, current_activity)

    def allows_event_relation(self, previous_activity, current_event):
        # Prüft eine Relation bei der Node des aktuellen Events.
        node = self.get_existing_node_for_event(current_event)
        if node is None:
            return False
        return node.allows(previous_activity, current_event.activity)

    @classmethod
    def from_event_log(cls, event_log):
        # Baut ein verteiltes Footprint-Modell aus Trainingsdaten

        from .builder import ModelBuilder

        return ModelBuilder().build(event_log)
