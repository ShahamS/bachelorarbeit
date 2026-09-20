from tokenbasedreplay.model.distributed.result import (
    ReplayEventResult,
    ReplayLogResult,
    ReplayTraceResult,
)
from tokenbasedreplay.model.participant.local_replayer import LocalReplayer
from tokenbasedreplay.model.distributed.network import NetworkSimulator
from tokenbasedreplay.model.participant.participant import Participant
class DistributedTokenReplayer:
    def __init__(self, discovered_model, participants, network_simulator: NetworkSimulator):
        self.discovered_model = discovered_model
        self.participants = participants
        self.network_simulator = network_simulator

    def replay_trace(self, case_id, trace):
        self.reset_case_markings(case_id)
        trace_result = ReplayTraceResult(case_id)
        local_replayers = {
            participant_id: LocalReplayer(
                self.discovered_model,
                participant,
                self.network_simulator,
            )
            for participant_id, participant in self.participants.items()
        }

        for event in trace:
            participant = self.network_simulator.route_event(event)
            if participant is None:
                trace_result.add_event_result(
                    ReplayEventResult(
                        case_id=case_id,
                        activity=event.activity,
                        participant_id=None,
                        matched=False,
                        unknown_activity=True,
                    )
                )
                continue

            consumed_before = participant.metrics.get("consumed_tokens", 0)
            produced_before = participant.metrics.get("produced_tokens", 0)
            missing_before = participant.metrics.get("missing_tokens", 0)
            local_replayer = local_replayers[participant.participant_id]
            try:
                local_replayer.replay(event.activity, case_id)
                unknown_activity = False
            except ValueError:
                unknown_activity = True

            consumed_after = participant.metrics.get("consumed_tokens", 0)
            produced_after = participant.metrics.get("produced_tokens", 0)
            missing_after = participant.metrics.get("missing_tokens", 0)
            consumed_tokens = consumed_after - consumed_before
            produced_tokens = produced_after - produced_before
            missing_tokens = missing_after - missing_before
            trace_result.add_event_result(
                ReplayEventResult(
                    case_id=case_id,
                    activity=event.activity,
                    participant_id=participant.participant_id,
                    matched=missing_tokens == 0 and not unknown_activity,
                    consumed_tokens=consumed_tokens,
                    produced_tokens=produced_tokens,
                    missing_tokens=missing_tokens,
                    unknown_activity=unknown_activity,
                )
            )
        self.resolve_silent_final_marking(case_id)
        self.finalize_case_markings(case_id)
        return trace_result

    def replay_log(self, event_log):
        log_result = ReplayLogResult()

        for case_id, trace in event_log.iter_traces():
            log_result.add_trace_result(self.replay_trace(case_id, trace))

        log_result.participant_metrics = {
            participant_id: participant.get_metrics()
            for participant_id, participant in self.participants.items()
        }
        log_result.network_metrics = self.network_simulator.metrics()

        return log_result

    def reset_case_markings(self, case_id):
        for participant in self.participants.values():
            participant.reset_case(case_id)
        for place_id, amount in self.discovered_model.initial_marking.items():
            owner = self.participant_for_place(place_id)
            owner.metrics["produced_tokens"] = (
                owner.metrics.get("produced_tokens", 0) + amount
            )

    def finalize_case_markings(self, case_id):
        marking = {}
        for participant in self.participants.values():
            for place_id, amount in participant.marking_for_case(case_id).tokens.items():
                marking[place_id] = marking.get(place_id, 0) + amount

        for place_id, amount in self.discovered_model.final_marking.items():
            owner = self.participant_for_place(place_id)
            owner.metrics["consumed_tokens"] = (
                owner.metrics.get("consumed_tokens", 0) + amount
            )

        for place_id, amount in marking.items():
            expected_amount = self.discovered_model.final_marking.get(place_id, 0)
            if amount > expected_amount:
                owner = self.participant_for_place(place_id)
                owner.metrics["remaining_tokens"] = (
                    owner.metrics.get("remaining_tokens", 0)
                    + amount
                    - expected_amount
                )

        for place_id, expected_amount in self.discovered_model.final_marking.items():
            current_amount = marking.get(place_id, 0)
            if current_amount < expected_amount:
                owner = self.participant_for_place(place_id)
                owner.metrics["missing_tokens"] = (
                    owner.metrics.get("missing_tokens", 0)
                    + expected_amount
                    - current_amount
                )

    def participant_for_place(self, place_id):
        participant_id = self.network_simulator.place_to_participant.get(place_id)
        if participant_id in self.participants:
            return self.participants[participant_id]
        return next(iter(self.participants.values()))

    def resolve_silent_final_marking(self, case_id):
        if not self.participants:
            return
        participant = next(iter(self.participants.values()))
        LocalReplayer(
            self.discovered_model,
            participant,
            self.network_simulator,
        ).resolve_silent_until_final_marking(case_id)
