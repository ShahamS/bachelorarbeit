from .distributed_model import DistributedFootprintModel
from checkmyflow.data.event_log import EventLog
from .footprint import END_ACTIVITY


class ModelBuilder:
    def build(self, training_log: EventLog):
        model = DistributedFootprintModel()
        for _, events in training_log.iter_traces():
            previous_event = None
            for current_event in events:
                model.add_event_relation(current_event, previous_event)
                previous_event = current_event
            if previous_event is not None:
                model.add_activity_relation(
                    previous_event.node_id,
                    previous_event.activity,
                    END_ACTIVITY,
                )
        return model
