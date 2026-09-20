from .footprint import START_ACTIVITY
from .node import Node


class DistributedFootprintModel:
    def __init__(self):
        self.nodes = {}
        self.activity_to_node = {}

    def get_or_create_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id=node_id)
        return self.nodes[node_id]

    def register_activity(self, activity, node_id):
        node = self.get_or_create_node(node_id)
        node.add_activity(activity)
        self.activity_to_node[activity] = node_id

    def add_event_relation(self, current_event, previous_event=None):
        current_node_id = current_event.node_id
        previous_activity = (
            START_ACTIVITY if previous_event is None else previous_event.activity
        )
        self.register_activity(current_event.activity, current_node_id)
        node = self.get_or_create_node(current_node_id)
        node.add_relation(previous_activity, current_event.activity)

    def add_activity_relation(self, node_id, previous_activity, current_activity):
        node = self.get_or_create_node(node_id)
        node.add_relation(previous_activity, current_activity, register_activity=False)

    def get_existing_node_for_event(self, event):
        return self.nodes.get(event.node_id)

    def get_node_for_activity(self, activity):
        node_id = self.activity_to_node.get(activity)
        if node_id is None:
            return None
        return self.nodes.get(node_id)

    def allows(self, previous_activity, current_activity):
        node = self.get_node_for_activity(current_activity)
        if node is None:
            return False
        return node.allows(previous_activity, current_activity)

    def allows_event_relation(self, previous_activity, current_event):
        node = self.get_existing_node_for_event(current_event)
        if node is None:
            return False
        return node.allows(previous_activity, current_event.activity)