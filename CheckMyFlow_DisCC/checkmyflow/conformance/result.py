class EventCheckResult:
    def __init__(
        self,
        case_id,
        event_index,
        previous_activity,
        current_activity,
        node_id,
        match,
        reason="match",
    ):
        self.case_id = case_id
        self.event_index = event_index
        self.previous_activity = previous_activity
        self.current_activity = current_activity
        self.node_id = node_id
        self.match = match
        self.reason = reason


class TraceCheckResult:
    def __init__(self, case_id, matches=0, mismatches=0, total_events=0):
        self.case_id = case_id
        self.matches = matches
        self.mismatches = mismatches
        self.total_events = total_events

    def add_event_result(self, event_result):
        self.total_events += 1
        if event_result.match:
            self.matches += 1
        else:
            self.mismatches += 1


class ConformanceResult:
    def __init__(self, event_results=None, trace_results=None):
        self.event_results = event_results or []
        self.trace_results = trace_results or {}
        self.route_calls = 0
        self.remote_calls = 0
        self.local_calls = 0

    def add_event_result(self, event_result):
        self.event_results.append(event_result)
        trace_result = self.trace_results.setdefault(
            event_result.case_id,
            TraceCheckResult(case_id=event_result.case_id),
        )
        trace_result.add_event_result(event_result)

    @property
    def total_matches(self):
        return sum(trace.matches for trace in self.trace_results.values())

    @property
    def total_mismatches(self):
        return sum(trace.mismatches for trace in self.trace_results.values())

    @property
    def total_events(self):
        return self.total_matches + self.total_mismatches

    @property
    def generalization(self):
        if self.total_events == 0:
            return 0.0
        return self.total_matches / self.total_events

    def reason_count(self, reason):
        return sum(1 for event_result in self.event_results if event_result.reason == reason)
