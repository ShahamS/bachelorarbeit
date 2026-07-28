from checkmyflow.model import DistributedFootprintModel, START_ACTIVITY

from .result import ConformanceResult, EventCheckResult


class OnlineChecker:
    '''Prüft Testtraces gegen ein bereits trainiertes Footprint-Modell.
    Unbekannte Aktivitäten, Nodes oder Vorgängerbeziehungen werden als Mismatch gewertet, damit die
    Testdaten nicht versehentlich in das Modell einfliessen.
    '''

    def __init__(self, model):
        self.model = model

    def check(self, event_log):
        '''Prüft alle Traces eines EventLogs in ihrer gespeicherten Reihenfolge.'''

        result = ConformanceResult()
        for case_id, events in event_log.iter_traces():
            previous_event = None
            for event_index, current_event in enumerate(events):
                # Prüfe Relation zwischen aktuellem Event und seinem direkten Vorgänger im selben Trace
                # Speichere das Ergebnis und aktualisiere die Trace-Aggregation
                result.add_event_result(
                    self.check_event(case_id, event_index, current_event, previous_event)
                )
                previous_event = current_event
        return result

    def check_event(self, case_id, event_index, current_event, previous_event):
        '''Prüft ein Event mit seinem direkten Vorgänger im selben Trace.'''

        previous_activity = (
            START_ACTIVITY if previous_event is None else previous_event.activity
        )
        # Prüfe, ob die Relation im Modell erlaubt ist, und ermittle den Grund für das Ergebnis
        match = self.model.allows_event_relation(previous_activity, current_event)
        reason = self._reason_for(previous_activity, current_event, match)

        return EventCheckResult(
            case_id=case_id,
            event_index=event_index,
            previous_activity=previous_activity,
            current_activity=current_event.activity,
            node_id=current_event.node_id,
            match=match,
            reason=reason,
        )

    def _reason_for(self, previous_activity, current_event, match):
        '''Erklärt das Ergebnis ohne das Modell beim Prüfen zu verändern.'''

        if match:
            return "match"

        node = self.model.get_existing_node_for_event(current_event)
        if node is None:
            return "unknown_node"
        return "invalid_predecessor"
