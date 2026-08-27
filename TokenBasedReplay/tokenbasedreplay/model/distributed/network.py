"""Simuliert Routing und Token-Anfragen zwischen Participants."""


class NetworkSimulator:
    def __init__(self, participant_mapping, activity_to_participant, place_to_participant=None):
        self.participant_mapping = participant_mapping
        self.activity_to_participant = activity_to_participant
        self.place_to_participant = place_to_participant or {}
        # Event routing wird seperat von der eigentlichen Tokenkommunikation gezählt
        self.route_calls = 0
        # Remote calls sind token requests zwischen Participants
        self.remote_calls = 0
        # Local calls sind Token-Checks die vom selben Participant ausgeführt werden
        self.local_token_checks = 0
        self.remote_token_checks = 0

    def route_event(self, event):
        # Dezentrales replay routet Event zum Participant via location
        participant_id = event.location

        if participant_id is None:
            # fallback for older tests or synthetic events.
            participant_id = self.activity_to_participant.get(event.activity)
            if participant_id is None:
                return None

        self.route_calls += 1
        return self.participant_mapping.get(participant_id)

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
