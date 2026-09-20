class NetworkSimulator:
    def __init__(self, participant_mapping, activity_to_participant, place_to_participant=None):
        self.participant_mapping = participant_mapping
        self.activity_to_participant = activity_to_participant
        self.place_to_participant = place_to_participant or {}
        self.transition_to_participant = self.index_transition_owners()
        self.route_calls = 0
        self.remote_calls = 0
        self.local_token_checks = 0
        self.remote_token_checks = 0

    def route_event(self, event):
        participant_id = event.location

        if participant_id is None:
            participant_id = self.activity_to_participant.get(event.activity)
            if participant_id is None:
                return None

        self.route_calls += 1
        return self.participant_mapping.get(participant_id)

    def index_transition_owners(self):
        owners = {}
        for participant_id, participant in self.participant_mapping.items():
            for transition in participant.transitions:
                owners.setdefault(transition, participant_id)
        return owners

    def transition_owner(self, transition, fallback_participant_id):
        return self.transition_to_participant.get(transition, fallback_participant_id)

    def request_token(self, requesting_participant_id, place_id, case_id=None):
        # fehlender Eigentümer oder gleicher Eigentümer Request ist lokaler Check
        # s. record_local_token_check
        owner_id = self.place_to_participant.get(place_id)
        if owner_id is None or owner_id == requesting_participant_id:
            self.local_token_checks += 1
            return False

        # Anderer Eigentümer bedeutet Kommunikationsaufwand zwischen Participants
        self.remote_calls += 1
        self.remote_token_checks += 1
        owner = self.participant_mapping.get(owner_id)
        if owner and owner.has_token(place_id, case_id):
            owner.consume_token(place_id, case_id)
            return True

        return False

    def snapshot_marking(self, case_id):
        marking = {}
        for participant in self.participant_mapping.values():
            for place_id, amount in participant.marking_for_case(case_id).tokens.items():
                marking[place_id] = marking.get(place_id, 0) + amount
        return marking

    def commit_transition(self, requesting_participant_id, transition, input_places, output_places, case_id):
        for place_id in input_places:
            owner_id = self.place_to_participant.get(place_id)
            owner = self.participant_mapping.get(owner_id)
            if owner_id is None or owner is None:
                continue
            if owner_id == requesting_participant_id:
                self.local_token_checks += 1
            else:
                self.remote_calls += 1
                self.remote_token_checks += 1
            owner.consume_token(place_id, case_id)

        for place_id in output_places:
            self.produce_token(requesting_participant_id, place_id, case_id)

    def produce_token(self, requesting_participant_id, place_id, case_id):
        owner_id = self.place_to_participant.get(place_id, requesting_participant_id)
        owner = self.participant_mapping.get(owner_id)
        if owner is None:
            return False

        owner.produce_token(place_id, case_id)
        return True

    def record_local_token_check(self):
        self.local_token_checks += 1

    def metrics(self):
        return {
            'route_calls': self.route_calls,
            'remote_calls': self.remote_calls,
            'local_calls': self.local_token_checks,
            'local_token_checks': self.local_token_checks,
            'remote_token_checks': self.remote_token_checks
        }
