from tokenbasedreplay.model.petrinet.net_access import PetriNetAccess
from tokenbasedreplay.model.petrinet.transition_lookup import TransitionLookup
'''
Führt eigentliche Token-based Replay pro Participant durch
'''

class LocalReplayer:
    def __init__(self, discovered_model, participant, network_simulator=None):
        self.discovered_model = discovered_model
        self.participant = participant
        self.network_simulator = network_simulator
        self.net_access = PetriNetAccess(discovered_model)
        self.transition_lookup = TransitionLookup(discovered_model.net)

    def replay(self, activity, case_id=None):
        # Transition für die gegebene Aktivität im Petri-Netz des Participants finden
        transition = self.transition_lookup.find_transition(activity)
        if transition is None:
            self._increment_metric('unknown_activities')
            raise ValueError(f"Activity {activity} existiert nicht im Petri-Netz.")

        # Prüfen, ob Transition im Petri-Netz des Participants existiert    
        if transition not in self.participant.transitions:
            raise ValueError(f"Transition {transition} existiert nicht im Petri-Netz des Participants.")

        input_places = self.net_access.input_places(transition)
        output_places = self.net_access.output_places(transition)

        # Prüfen, ob alle Input-Places Tokens haben; wenn nicht, fehlende Tokens zählen
        for place_id in input_places:
            self._increment_metric('consumed_tokens')
            if self.participant.has_token(place_id, case_id):
                self._record_local_token_check()
                self.participant.consume_token(place_id, case_id)
            elif self._request_remote_token(place_id, case_id):
                # Schau, ob ein anderer Participant Eigentümer vom Place ist und Token besitzt
                continue
            else:
                # Niemand besitzt Token, also fehlt Token zum Feuern
                self._increment_metric('missing_tokens')

        # Tokens in Output-Places produzieren
        for place_id in output_places:
            self.participant.produce_token(place_id, case_id)
            self._increment_metric('produced_tokens')

    def _increment_metric(self, metric_name):
        self.participant.metrics[metric_name] = self.participant.metrics.get(metric_name, 0) + 1

    def _request_remote_token(self, place_id, case_id=None):
        if self.network_simulator is None:
            return False
        return self.network_simulator.request_token(
            self.participant.participant_id,
            place_id,
            case_id,
        )

    def _record_local_token_check(self):
        if self.network_simulator is not None:
            self.network_simulator.record_local_token_check()
