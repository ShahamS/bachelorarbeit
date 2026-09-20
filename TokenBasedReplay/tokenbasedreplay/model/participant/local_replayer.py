from collections import deque

from tokenbasedreplay.model.petrinet.net_access import PetriNetAccess
from tokenbasedreplay.model.petrinet.transition_lookup import TransitionLookup

class LocalReplayer:
    def __init__(
        self,
        discovered_model,
        participant,
        network_simulator=None,
        max_silent_depth=100,
        max_silent_states=1000,
    ):
        self.discovered_model = discovered_model
        self.participant = participant
        self.network_simulator = network_simulator
        self.net_access = PetriNetAccess(discovered_model)
        self.transition_lookup = TransitionLookup(discovered_model.net)
        self.max_silent_depth = max_silent_depth
        self.max_silent_states = max_silent_states

    def replay(self, activity, case_id):
        transition = self.transition_lookup.find_transition(activity)
        if transition is None:
            self.increment_metric('unknown_activities')
            raise ValueError(f"Activity {activity} existiert nicht im Petri-Netz.")

        if transition not in self.participant.transitions:
            raise ValueError(f"Transition {transition} existiert nicht im Petri-Netz des Participants.")

        input_places = self.net_access.input_places(transition)
        output_places = self.net_access.output_places(transition)

        if not self.has_all_input_tokens(input_places, case_id):
            missing_places = self.missing_places(input_places, case_id)
            self.resolve_silent_until_transition_enabled(transition, missing_places, case_id)

        for place_id in input_places:
            self.increment_metric('consumed_tokens')
            if self.participant.has_token(place_id, case_id):
                self.record_local_token_check()
                self.participant.consume_token(place_id, case_id)
            elif self.request_remote_token(place_id, case_id):
                continue
            else:
                self.increment_metric('missing_tokens')

        for place_id in output_places:
            self.produce_token(place_id, case_id)
            self.increment_metric('produced_tokens')

    def increment_metric(self, metric_name):
        self.participant.metrics[metric_name] = self.participant.metrics.get(metric_name, 0) + 1

    def request_remote_token(self, place_id, case_id):
        if self.network_simulator is None:
            return False
        return self.network_simulator.request_token(
            self.participant.participant_id,
            place_id,
            case_id,
        )

    def record_local_token_check(self):
        if self.network_simulator is not None:
            self.network_simulator.record_local_token_check()

    def produce_token(self, place_id, case_id):
        if self.network_simulator is None:
            self.participant.produce_token(place_id, case_id)
            return

        produced = self.network_simulator.produce_token(
            self.participant.participant_id,
            place_id,
            case_id,
        )
        if not produced:
            self.participant.produce_token(place_id, case_id)

    def resolve_silent_until_final_marking(self, case_id):
        if self.network_simulator is None:
            return False
        marking = self.network_simulator.snapshot_marking(case_id)
        missing_places = [
            place_id
            for place_id, expected_amount in self.discovered_model.final_marking.items()
            if marking.get(place_id, 0) < expected_amount
        ]
        return self.resolve_silent_until(
            lambda marking: self.satisfies_final_marking(marking),
            missing_places,
            case_id,
        )

    def resolve_silent_until_transition_enabled(self, transition, missing_places, case_id):
        if self.network_simulator is None:
            return False
        return self.resolve_silent_until(
            lambda marking: self.is_enabled_in_marking(transition, marking),
            missing_places,
            case_id,
        )

    def resolve_silent_until(self, target_reached, missing_places, case_id):
        start_marking = self.network_simulator.snapshot_marking(case_id)
        silent_transitions = self.relevant_silent_transitions(missing_places, start_marking)
        if not silent_transitions:
            return False

        stack = [(dict(start_marking), [])]
        visited = {self.marking_key(start_marking)}

        while stack:
            marking, path = stack.pop()
            if target_reached(marking):
                self.commit_silent_path(path, case_id)
                return True
            if len(path) >= self.max_silent_depth:
                continue
            if len(visited) >= self.max_silent_states:
                return False

            for transition in reversed(silent_transitions):
                if not self.is_enabled_in_marking(transition, marking):
                    continue
                next_marking = self.fire_on_copy(marking, transition)
                key = self.marking_key(next_marking)
                if key in visited:
                    continue
                visited.add(key)
                stack.append((next_marking, path + [transition]))

        return False

    def missing_places(self, input_places, case_id):
        if self.network_simulator is None:
            marking = self.participant.marking_for_case(case_id).tokens
        else:
            marking = self.network_simulator.snapshot_marking(case_id)
        return [place_id for place_id in input_places if marking.get(place_id, 0) <= 0]

    def relevant_silent_transitions(self, target_places, marking=None):
        backward_relevant = self.backward_relevant_silent_transitions(target_places)
        if marking is None:
            return backward_relevant

        forward_reachable = self.forward_reachable_silent_transitions(marking)
        if not forward_reachable:
            return []

        return [
            transition
            for transition in backward_relevant
            if transition in forward_reachable
        ]

    def backward_relevant_silent_transitions(self, target_places):
        queue = deque(target_places)
        visited_places = set(target_places)
        relevant = []
        relevant_seen = set()

        while queue:
            place = queue.popleft()
            for arc in getattr(place, "in_arcs", []):
                transition = arc.source
                if not self.net_access.is_silent(transition):
                    continue
                if transition not in relevant_seen:
                    relevant_seen.add(transition)
                    relevant.append(transition)

                for input_place in self.net_access.input_places(transition):
                    if input_place in visited_places:
                        continue
                    visited_places.add(input_place)
                    queue.append(input_place)

        return relevant

    def forward_reachable_silent_transitions(self, marking):
        queue = deque(
            place
            for place, amount in marking.items()
            if amount > 0
        )
        visited_places = set(queue)
        reachable = set()

        while queue:
            place = queue.popleft()
            for arc in getattr(place, "out_arcs", []):
                transition = arc.target
                if not self.net_access.is_silent(transition):
                    continue
                if transition in reachable:
                    continue
                reachable.add(transition)

                for output_place in self.net_access.output_places(transition):
                    if output_place in visited_places:
                        continue
                    visited_places.add(output_place)
                    queue.append(output_place)

        return reachable

    def commit_silent_path(self, path, case_id):
        for transition in path:
            participant_id = self.participant.participant_id
            if self.network_simulator is not None:
                participant_id = self.network_simulator.transition_owner(
                    transition,
                    participant_id,
                )
            self.network_simulator.commit_transition(
                participant_id,
                transition,
                self.net_access.input_places(transition),
                self.net_access.output_places(transition),
                case_id,
            )

    def has_all_input_tokens(self, input_places, case_id):
        if self.network_simulator is not None:
            marking = self.network_simulator.snapshot_marking(case_id)
            return all(marking.get(place_id, 0) > 0 for place_id in input_places)
        return all(self.participant.has_token(place_id, case_id) for place_id in input_places)

    def is_enabled_in_marking(self, transition, marking):
        return all(
            marking.get(place_id, 0) > 0
            for place_id in self.net_access.input_places(transition)
        )

    def fire_on_copy(self, marking, transition):
        next_marking = dict(marking)
        for place_id in self.net_access.input_places(transition):
            next_marking[place_id] = next_marking.get(place_id, 0) - 1
            if next_marking[place_id] <= 0:
                next_marking.pop(place_id, None)
        for place_id in self.net_access.output_places(transition):
            next_marking[place_id] = next_marking.get(place_id, 0) + 1
        return next_marking

    def satisfies_final_marking(self, marking):
        return all(
            marking.get(place_id, 0) >= expected_amount
            for place_id, expected_amount in self.discovered_model.final_marking.items()
        )

    def marking_key(self, marking):
        return tuple(sorted((str(place_id), amount) for place_id, amount in marking.items()))
