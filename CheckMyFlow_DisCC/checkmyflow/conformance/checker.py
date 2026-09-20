from checkmyflow.model import DistributedFootprintModel, END_ACTIVITY, START_ACTIVITY

from .result import ConformanceResult, EventCheckResult


class OnlineChecker:
    def __init__(self, model: DistributedFootprintModel):
        self.model = model

    def check(self, event_log):
        result = ConformanceResult()
        # Speicher für Events ohne Nachfolger
        node_states = {
            node_id: {}
            for node_id in self.model.nodes
        }
        for case_id, events in event_log.iter_traces():
            for event_index, current_event in enumerate(events):
                previous_event = self.find_previous_event(
                    case_id,
                    current_event,
                    node_states,
                    result,
                )
                result.add_event_result(
                    self.check_event(case_id, event_index, current_event, previous_event)
                )
                self.store_open_event(current_event, node_states)
            self.check_trace_end(case_id, len(events), node_states, result)
        return result

    def find_previous_event(self, case_id, current_event, node_states, result):
        result.route_calls += 1
        current_node_id = current_event.node_id
        candidates = []

        current_node_state = node_states.get(current_node_id, {})
        if case_id in current_node_state:
            # Knoten, bei dem derzeitiges Event auftaucht, hat selber noch einen potentiellen Vorgänger offen
            result.local_calls += 1
            candidates.append((current_node_id, current_node_state[case_id]))

        for node_id, node_state in node_states.items():
            if node_id == current_node_id:
                continue
            # Request an anderen Knoten
            result.remote_calls += 1
            if case_id not in node_state:
                continue
            # Response vom anderen Knoten
            result.remote_calls += 1
            candidates.append((node_id, node_state[case_id]))

        if not candidates:
            return None

        predecessor_node_id, previous_event = max(
            candidates,
            key=lambda item: item[1].time,
        )
        self.close_predecessor(
            case_id,
            current_node_id,
            predecessor_node_id,
            node_states,
            result,
        )
        return previous_event

    def close_predecessor(
        self,
        case_id,
        current_node_id,
        predecessor_node_id,
        node_states,
        result,
    ):
        if predecessor_node_id == current_node_id:
            # Lokales Event ist der Vorgänger
            result.local_calls += 1
        else:
            # Anderen Knoten informieren, dass er den Vorgängerevent besitzt
            result.remote_calls += 1
        node_states[predecessor_node_id].pop(case_id, None)

    def store_open_event(self, event, node_states):
        node_states.setdefault(event.node_id, {})[event.case_id] = event

    def check_trace_end(self, case_id, event_index, node_states, result):
        candidates = [
            event
            for node_state in node_states.values()
            if case_id in node_state
            for event in [node_state[case_id]]
        ]
        if not candidates:
            return

        previous_event = max(candidates, key=lambda event: event.time)
        node = self.model.get_existing_node_for_event(previous_event)
        match = node is not None and node.allows(previous_event.activity, END_ACTIVITY)
        result.add_event_result(
            EventCheckResult(
                case_id=case_id,
                event_index=event_index,
                previous_activity=previous_event.activity,
                current_activity=END_ACTIVITY,
                node_id=previous_event.node_id,
                match=match,
                reason="match" if match else "invalid_end",
            )
        )
        node_states[previous_event.node_id].pop(case_id, None)

    def check_event(self, case_id, event_index, current_event, previous_event):
        previous_activity = (
            START_ACTIVITY if previous_event is None else previous_event.activity
        )
        match = self.model.allows_event_relation(previous_activity, current_event)
        reason = self.reason_for(previous_activity, current_event, match)

        return EventCheckResult(
            case_id=case_id,
            event_index=event_index,
            previous_activity=previous_activity,
            current_activity=current_event.activity,
            node_id=current_event.node_id,
            match=match,
            reason=reason,
        )

    def reason_for(self, previous_activity, current_event, match):
        if match:
            return "match"

        node = self.model.get_existing_node_for_event(current_event)
        if node is None:
            return "unknown_node"
        return "invalid_predecessor"
