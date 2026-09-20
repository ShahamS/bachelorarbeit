from tokenbasedreplay.model.petrinet.marking import LocalMarking

class Participant:
    def __init__(self, participant_id):
        self.participant_id = participant_id
        self.places = []
        self.transitions = []
        self.initial_marking = LocalMarking()
        self.case_markings = {}
        self.metrics = {}

    def set_initial_token(self, place_id, amount=1):
        self.initial_marking.add(place_id, amount)
        
    def reset_case(self, case_id):
        marking = LocalMarking() 
        for place_id, amount in self.initial_marking.tokens.items():
            marking.add(place_id, amount)
        self.case_markings[case_id] = marking

    def marking_for_case(self, case_id):
        if case_id not in self.case_markings:
            self.reset_case(case_id)
        return self.case_markings[case_id]

    def has_token(self, place_id, case_id):
        return self.marking_for_case(case_id).get(place_id) > 0

    def consume_token(self, place_id, case_id):
        self.marking_for_case(case_id).remove(place_id)

    def produce_token(self, place_id, case_id):
        self.marking_for_case(case_id).add(place_id)

    def get_metrics(self):
        return self.metrics
