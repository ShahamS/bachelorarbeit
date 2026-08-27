from tokenbasedreplay.model.petrinet.marking import LocalMarking
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
        self.local_marking = LocalMarking()
        self.initial_marking = LocalMarking()
        self.case_markings = {}
        self.metrics = {}

    def set_initial_token(self, place_id, amount=1):
        # Initial tokens define the reproducible start state for every new case.
        self.initial_marking.add(place_id, amount)
        self.local_marking.add(place_id, amount)

    def reset_case(self, case_id):
        # Online replay hat ein Marking pro Case => kopiere initial Marking
        marking = LocalMarking() 
        for place_id, amount in self.initial_marking.tokens.items():
            marking.add(place_id, amount)
        self.case_markings[case_id] = marking

    def marking_for_case(self, case_id=None):
        if case_id is None:
            return self.local_marking
        if case_id not in self.case_markings:
            self.reset_case(case_id)
        return self.case_markings[case_id]

    def has_token(self, place_id, case_id=None):
        return self.marking_for_case(case_id).get(place_id) > 0

    def consume_token(self, place_id, case_id=None):
        self.marking_for_case(case_id).remove(place_id)

    def produce_token(self, place_id, case_id=None):
        self.marking_for_case(case_id).add(place_id)

    def finalize_remaining(self, final_marking=None, case_id=None):
        marking = self.marking_for_case(case_id)
        # Berechne übrige Tokens durch Vergleich zwischen Finales Marking lt. Discovery
        # vs. aktuelles Marking des Cases
        if final_marking is None:
            remaining_tokens = marking.remaining_tokens()
            self.metrics['remaining_tokens'] = self.metrics.get('remaining_tokens', 0) + remaining_tokens
            return
    
        remaining_tokens = 0
        for place_id, amount in marking.tokens.items():
            expected_amount = final_marking.get(place_id, 0)
            if amount > expected_amount:
                remaining_tokens += amount - expected_amount
        self.metrics['remaining_tokens'] = self.metrics.get('remaining_tokens', 0) + remaining_tokens

    def get_metrics(self):
        return self.metrics
