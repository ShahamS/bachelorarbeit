from .distributed_model import DistributedFootprintModel


class ModelBuilder:
    #Baut ein DistributedFootprintModel aus Trainings-Traces
    def build(self, training_log):
        #Trainiert ein neues Modell ausschliesslich auf dem übergebenen Log
        model = DistributedFootprintModel()
        for _, events in training_log.iter_traces():
            previous_event = None
            # Füge Relationen zwischen aufeinanderfolgenden Events hinzu, um die Footprintmatrix zu erstellen
            for current_event in events:
                model.add_event_relation(current_event, previous_event)
                previous_event = current_event
        return model
