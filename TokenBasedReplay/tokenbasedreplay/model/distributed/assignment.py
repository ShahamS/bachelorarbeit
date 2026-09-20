from collections import deque

from tokenbasedreplay.data.event_log import EventLog
from tokenbasedreplay.model.participant.participant import Participant
from tokenbasedreplay.model.petrinet.discovered_model import DiscoveredPetriNet
from tokenbasedreplay.model.petrinet.net_access import PetriNetAccess
from tokenbasedreplay.model.petrinet.transition_lookup import TransitionLookup

class ParticipantAssignment:
    def __init__(self, discovered_model: DiscoveredPetriNet, training_log: EventLog):
        self.discovered_model = discovered_model
        self.training_log = training_log
        self.participant_mapping = {}  
        self.activity_to_participant = {} 
        self.place_to_participant = {}  

    def build(self):
        net_access = PetriNetAccess(self.discovered_model)
        transition_lookup = TransitionLookup(self.discovered_model.net)

        for _, events in self.training_log.iter_traces():
            for event in events:
                activity = event.activity
                location = event.location

                if location is None:
                    raise ValueError(f"Event {event} hat keine Participant-Location.")
    
                transition = transition_lookup.find_transition(activity)
                if transition is not None:
                    participant = self.participant_for(location)
                    self.add_unique(participant.transitions, transition)
                    for place in net_access.input_places(transition):
                        self.assign_place(participant, place)
                    for place in net_access.output_places(transition):
                        self.assign_place(participant, place)
                    self.participant_mapping[location] = participant
                    self.activity_to_participant[activity] = location

        self.assign_silent_transitions(net_access, transition_lookup)
        self.assign_unowned_places(net_access)
        self.assign_initial_marking()

        return self.participant_mapping, self.activity_to_participant, self.place_to_participant

    def participant_for(self, participant_id):
        if participant_id not in self.participant_mapping:
            self.participant_mapping[participant_id] = Participant(participant_id)
        return self.participant_mapping[participant_id]

    def assign_place(self, participant, place):
        self.add_unique(participant.places, place)
        self.place_to_participant.setdefault(place, participant.participant_id)

    def assign_silent_transitions(self, net_access, transition_lookup):
        for transition in transition_lookup.silent_transitions():
            participant = self.nearest_participant_for_transition(transition, net_access)
            if participant is None:
                continue

            self.add_unique(participant.transitions, transition)
            for place in net_access.input_places(transition):
                if place not in self.place_to_participant:
                    self.assign_place(participant, place)
            for place in net_access.output_places(transition):
                if place not in self.place_to_participant:
                    self.assign_place(participant, place)

    def assign_unowned_places(self, net_access):
        for place in sorted(net_access.places(), key=lambda value: str(value.name)):
            if place in self.place_to_participant:
                continue
            participant = self.nearest_participant_for_place(place)
            if participant is not None:
                self.assign_place(participant, place)

    def nearest_participant_for_transition(self, transition, net_access):
        input_candidate = self.nearest_participant_from_places(net_access.input_places(transition))
        output_candidate = self.nearest_participant_from_places(net_access.output_places(transition))
        return self.choose_candidate(input_candidate, output_candidate)

    def nearest_participant_for_place(self, place):
        candidate = self.nearest_participant_from_places([place])
        if candidate is None:
            return None
        return candidate[1]

    def nearest_participant_from_places(self, start_places):
        queue = deque((place, 0) for place in start_places)
        visited = set(start_places)
        best = None

        while queue:
            node, distance = queue.popleft()
            if node in self.place_to_participant:
                participant_id = self.place_to_participant[node]
                candidate = (distance, self.participant_mapping[participant_id])
                if best is None or self.candidate_key(candidate) < self.candidate_key(best):
                    best = candidate
                continue

            for neighbor in self.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        return best

    def neighbors(self, node):
        neighbors = []
        for arc in getattr(node, "in_arcs", []):
            neighbors.append(arc.source)
        for arc in getattr(node, "out_arcs", []):
            neighbors.append(arc.target)
        return sorted(neighbors, key=lambda value: str(value.name))

    def choose_candidate(self, input_candidate, output_candidate):
        if input_candidate is None and output_candidate is None:
            return None
        if input_candidate is None:
            return output_candidate[1]
        if output_candidate is None:
            return input_candidate[1]
        if self.candidate_key(input_candidate) <= self.candidate_key(output_candidate):
            return input_candidate[1]
        return output_candidate[1]

    def candidate_key(self, candidate):
        distance, participant = candidate
        return distance, str(participant.participant_id)

    def assign_initial_marking(self):
        for place, amount in self.discovered_model.initial_marking.items():
            participant_id = self.place_to_participant.get(place)
            if participant_id is None:
                continue
            self.participant_mapping[participant_id].set_initial_token(place, amount)

    def add_unique(self, values, value):
        if value not in values:
            values.append(value)
