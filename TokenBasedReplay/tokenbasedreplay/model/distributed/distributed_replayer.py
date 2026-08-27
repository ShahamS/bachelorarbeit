"""
Führt Token-Based Replay über ganze Traces und Logs aus.
"""

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
        # Jeder Trace repräsentiert ein Online Case und bekommt sein eigenes Marking
        self._reset_case_markings(case_id)
        trace_result = ReplayTraceResult(case_id)

        for event in trace:
            # Network Simulator entscheidet welcher Participant das eingehende Event empfängt
            participant = self.network_simulator.route_event(event)
            if participant is None:
                # Es gibt bisher kein Participant, das für dieses Event zuständig ist
                # Zähle das als unknown activity
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
            # Replay ist lokal, aber Token requests könnten zwischen versch. Participants entstehen
            local_replayer = LocalReplayer(
                self.discovered_model,
                participant,
                self.network_simulator,
            )
            # Führe Replay lokal beim Participant aus
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
            # Sammle einzelne Ergebnisse des Events im Trace Ergebnis
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
        # Berechne finales Ergebnis vom Trace
        self._finalize_case_markings(case_id)
        return trace_result

    def replay_log(self, event_log):
        log_result = ReplayLogResult()

        for case_id, trace in event_log.iter_traces():
            # Sammle alle Trace Results, um komplettes Log Ergebnis zu bekommen
            log_result.add_trace_result(self.replay_trace(case_id, trace))

        log_result.participant_metrics = {
            participant_id: participant.get_metrics()
            for participant_id, participant in self.participants.items()
        }
        log_result.network_metrics = self.network_simulator.metrics()

        return log_result

    def _reset_case_markings(self, case_id):
        for participant in self.participants.values():
            participant.reset_case(case_id)

    def _finalize_case_markings(self, case_id):
        for participant in self.participants.values():
            participant.finalize_remaining(self.discovered_model.final_marking, case_id)
