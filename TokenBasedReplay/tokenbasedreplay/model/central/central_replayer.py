"""Zentrale Baseline fuer Token-Based Replay."""

from tokenbasedreplay.data.event_log import EventLog
from tokenbasedreplay.model.central.result import (
    CentralReplayEventResult,
    CentralReplayLogResult,
    CentralReplayTraceResult,
)
from tokenbasedreplay.model.petrinet.discovered_model import DiscoveredPetriNet
from tokenbasedreplay.model.petrinet.marking import LocalMarking
from tokenbasedreplay.model.petrinet.net_access import PetriNetAccess
from tokenbasedreplay.model.petrinet.transition_lookup import TransitionLookup

class CentralTokenReplayer:

    def __init__(self, discovered_model: DiscoveredPetriNet):
        self.discovered_model = discovered_model
        self.transition_lookup = TransitionLookup(discovered_model.net)
        self.net_access = PetriNetAccess(discovered_model)

    def replay_log(self, log: EventLog):
        log_result = CentralReplayLogResult()
        for case_id, trace in log.iter_traces():
            trace_result = self.replay_trace(case_id, trace)
            log_result.add_trace_result(trace_result)
        return log_result
    
    def replay_trace(self, case_id, trace):
        marking = self._initial_marking()
        trace_result = CentralReplayTraceResult(case_id)

        for event in trace:
            transition = self.transition_lookup.find_transition(event.activity)
            if transition is None:
                trace_result.add_event_result(
                    CentralReplayEventResult(
                        case_id=case_id,
                        activity=event.activity,
                        matched=False,
                        unknown_activity=True,
                    )
                )
                continue

            event_result = CentralReplayEventResult(
                case_id=case_id,
                activity=event.activity,
                matched=True,
            )
            input_places = self.net_access.input_places(transition)
            output_places = self.net_access.output_places(transition)

            for place_id in input_places:
                event_result.consumed_tokens += 1
                if marking.get(place_id) == 0:
                    event_result.missing_tokens += 1
                    event_result.matched = False
                else:
                    marking.remove(place_id)

            for place_id in output_places:
                marking.add(place_id)
                event_result.produced_tokens += 1

            trace_result.add_event_result(event_result)

        trace_result.remaining_tokens = self._remaining_tokens(marking)
        trace_result.missing_final_tokens = self._missing_final_tokens(marking)
        return trace_result

    def _initial_marking(self):
        marking = LocalMarking()
        for place, amount in self.discovered_model.initial_marking.items():
            marking.add(place, amount)
        return marking

    def _remaining_tokens(self, marking):
        remaining = 0
        for place, amount in marking.tokens.items():
            expected_amount = self.discovered_model.final_marking.get(place, 0)
            if amount > expected_amount:
                remaining += amount - expected_amount
        return remaining

    def _missing_final_tokens(self, marking):
        missing = 0
        for place, expected_amount in self.discovered_model.final_marking.items():
            current_amount = marking.get(place)
            if current_amount < expected_amount:
                missing += expected_amount - current_amount
        return missing
