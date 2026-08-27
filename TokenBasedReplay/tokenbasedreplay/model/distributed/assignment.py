"""Erzeugt Participants und Routing-Zuordnungen fuer verteiltes Replay."""

from tokenbasedreplay.data.event_log import EventLog
from tokenbasedreplay.model.participant.participant import Participant
from tokenbasedreplay.model.petrinet.discovered_model import DiscoveredPetriNet
from tokenbasedreplay.model.petrinet.net_access import PetriNetAccess
from tokenbasedreplay.model.petrinet.transition_lookup import TransitionLookup

class ParticipantAssignment:
    def __init__(self, discovered_model: DiscoveredPetriNet, training_log: EventLog):
        self.discovered_model = discovered_model
        self.training_log = training_log
        self.participant_mapping = {}  # Mapping von Locations zu Participants
        self.activity_to_participant = {}  # Mapping von Aktivität zu Participant-ID
        self.place_to_participant = {}  # Mapping von Place zu besitzendem Participant

    def build(self):
        # Baut Participants und gibt die Zuordnungen explizit zurück

        net_access = PetriNetAccess(self.discovered_model)
        transition_lookup = TransitionLookup(self.discovered_model.net)

        for case_id, events in self.training_log.iter_traces():
            for event in events:
                activity = event.activity
                location = event.location

                if location is None:
                    raise ValueError(f"Event {event} hat keine Participant-Location.")
                # Suche nach Transition via Activity
                transition = transition_lookup.find_transition(activity)
                if transition is not None:
                    # Suche nach Participant via Location oder erzeuge neuen Participant
                    participant = self._participant_for(location)
                    # Füge dem Participant die Transition zu
                    self._add_unique(participant.transitions, transition)
                    # Füge dem Participant Input- und Output-Places hinzu
                    for place in net_access.input_places(transition):
                        self._assign_place(participant, place)
                    for place in net_access.output_places(transition):
                        self._assign_place(participant, place)

                    # Speicher Participant ab und mappe
                    self.participant_mapping[location] = participant
                    self.activity_to_participant[activity] = location

        self._assign_initial_marking()

        return self.participant_mapping, self.activity_to_participant, self.place_to_participant

    def _participant_for(self, participant_id):
        if participant_id not in self.participant_mapping:
            self.participant_mapping[participant_id] = Participant(participant_id)
        return self.participant_mapping[participant_id]

    def _assign_place(self, participant, place):
        self._add_unique(participant.places, place)
        self.place_to_participant.setdefault(place, participant.participant_id)

    def _assign_initial_marking(self):
        for place, amount in self.discovered_model.initial_marking.items():
            participant_id = self.place_to_participant.get(place)
            if participant_id is None:
                continue
            # Discovery is central, but the initial marking is copied to the
            # owning participant so every online case can start reproducibly.
            self.participant_mapping[participant_id].set_initial_token(place, amount)

    def _add_unique(self, values, value):
        if value not in values:
            values.append(value)
