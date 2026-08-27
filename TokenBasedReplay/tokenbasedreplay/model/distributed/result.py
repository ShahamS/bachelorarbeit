"""
Ergebnisobjekte für verteiltes Token-Based Replay.
"""

from dataclasses import dataclass, field


@dataclass
class ReplayEventResult:
    case_id: str
    activity: str
    participant_id: str | None
    matched: bool
    consumed_tokens: int = 0
    produced_tokens: int = 0
    missing_tokens: int = 0
    unknown_activity: bool = False


@dataclass
class ReplayTraceResult:
    case_id: str
    events: list[ReplayEventResult] = field(default_factory=list)

    def add_event_result(self, event_result):
        self.events.append(event_result)

    @property
    def consumed_tokens(self):
        return sum(event.consumed_tokens for event in self.events)

    @property
    def produced_tokens(self):
        return sum(event.produced_tokens for event in self.events)

    @property
    def missing_tokens(self):
        return sum(event.missing_tokens for event in self.events)


@dataclass
class ReplayLogResult:
    traces: list[ReplayTraceResult] = field(default_factory=list)
    participant_metrics: dict = field(default_factory=dict)
    network_metrics: dict = field(default_factory=dict)

    def add_trace_result(self, trace_result):
        self.traces.append(trace_result)

    @property
    def total_events(self):
        return sum(len(trace.events) for trace in self.traces)

    @property
    def missing_tokens(self):
        return sum(event.missing_tokens for trace in self.traces for event in trace.events)

    @property
    def total_consumed_tokens(self):
        return self._sum_participant_metric("consumed_tokens")

    @property
    def total_produced_tokens(self):
        return self._sum_participant_metric("produced_tokens")

    @property
    def total_missing_tokens(self):
        return self._sum_participant_metric("missing_tokens")

    @property
    def total_remaining_tokens(self):
        return self._sum_participant_metric("remaining_tokens")

    @property
    def fitness(self):
        if self.total_consumed_tokens == 0 and self.total_produced_tokens == 0:
            return 0.0

        missing_component = 1.0
        if self.total_consumed_tokens > 0:
            missing_component = 1 - self.total_missing_tokens / self.total_consumed_tokens

        remaining_component = 1.0
        if self.total_produced_tokens > 0:
            remaining_component = 1 - self.total_remaining_tokens / self.total_produced_tokens

        return 0.5 * missing_component + 0.5 * remaining_component

    @property
    def unknown_activities(self):
        return sum(1 for trace in self.traces for event in trace.events if event.unknown_activity)

    @property
    def matched_events(self):
        return sum(1 for trace in self.traces for event in trace.events if event.matched)

    @property
    def route_calls(self):
        return self.network_metrics.get("route_calls", 0)

    @property
    def remote_calls(self):
        return self.network_metrics.get("remote_calls", 0)

    @property
    def local_calls(self):
        return self.network_metrics.get("local_calls", 0)

    def _sum_participant_metric(self, metric_name):
        return sum(
            metrics.get(metric_name, 0)
            for metrics in self.participant_metrics.values()
        )
