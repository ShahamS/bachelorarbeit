'''
Repräsentiert einen Teilnehmer in einem Prozessmodell, der durch ein 
Petri-Netz beschrieben wird. Hierdurch kann der Token-Based Replay Ansatz 
dezentralisiert auf mehrere Teilnehmer angewendet werden.
'''
class Participant:
    def __init__(self, participant_id):
        self.participant_id = participant_id
        self.places = []
        self.transitions = []
        self.local_marking = {}
        self.metrics = {}

    def has_token(self, place_id):
        pass

    def consume_token(self, place_id):
        pass

    def produce_token(self, place_id):
        pass

    def finalize_remaining(self):
        pass

    def get_metrics(self):
        return self.metrics
