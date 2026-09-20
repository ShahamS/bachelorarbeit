import time

from checkmyflow.conformance import OnlineChecker
from checkmyflow.model import ModelBuilder


class EvaluationSummary:
    def __init__(self, metrics):
        self.metrics = metrics

    def to_dict(self):
        return dict(self.metrics)

def evaluate_checkmyflow(training_log, test_log):
    training_start = time.perf_counter()
    model = ModelBuilder().build(training_log)
    training_time_s = time.perf_counter() - training_start

    checking_start = time.perf_counter()
    result = OnlineChecker(model).check(test_log)
    checking_time_s = time.perf_counter() - checking_start
    fitness_result = OnlineChecker(model).check(training_log)

    model_metrics = _model_metrics(model)
    result_metrics = _result_metrics(result)
    quality_metrics = _quality_metrics(
        fitness=fitness_result.generalization,
        generalization=result.generalization,
    )
    network_metrics = _network_metrics(result)
    timing_metrics = _timing_metrics(training_time_s, checking_time_s, result)

    metrics = {}
    metrics.update(result_metrics)
    metrics.update(quality_metrics)
    metrics.update(network_metrics)
    metrics.update(model_metrics)
    metrics.update(timing_metrics)

    return EvaluationSummary(metrics)


def _result_metrics(result):
    total_events = result.total_events
    mismatches = result.total_mismatches
    matches = result.total_matches

    if total_events == 0:
        exact_pct = 0.0
    else:
        exact_pct = matches / total_events

    return {
        "total_events": total_events,
        "matches": matches,
        "mismatches": mismatches,
        "generalization": result.generalization,
        "exact_pct": exact_pct,
        "unknown_node": result.reason_count("unknown_node"),
        "invalid_predecessor": result.reason_count("invalid_predecessor"),
        "invalid_end": result.reason_count("invalid_end"),
    }


def _quality_metrics(fitness, generalization):
    return {
        "fitness": fitness,
        "fitness_generalization_gap": fitness - generalization,
    }


def _network_metrics(result):
    total_events = result.total_events
    return {
        "route_calls": result.route_calls,
        "remote_calls": result.remote_calls,
        "local_calls": result.local_calls,
        "requests_per_event": 0.0 if total_events == 0 else result.remote_calls / total_events,
    }


def _model_metrics(model):
    relation_count = 0
    activity_count = 0
    max_node_relations = 0

    for node in model.nodes.values():
        node_relations = 0
        activity_count += len(node.activities)
        for predecessors in node.footprint_matrix.allowed_predecessors.values():
            node_relations += len(predecessors)
        relation_count += node_relations
        max_node_relations = max(max_node_relations, node_relations)

    return {
        "nodes": len(model.nodes),
        "activities": activity_count,
        "matrix_entries": relation_count,
        "max_node_matrix_entries": max_node_relations,
    }


def _timing_metrics(training_time_s, checking_time_s, result):
    total_events = result.total_events
    elapsed_s = training_time_s + checking_time_s

    if total_events == 0:
        avg_event_time_ms = 0.0
    else:
        avg_event_time_ms = (checking_time_s / total_events) * 1000

    return {
        "training_time_s": training_time_s,
        "checking_time_s": checking_time_s,
        "elapsed_s": elapsed_s,
        "avg_event_time_ms": avg_event_time_ms,
        "total_compute": total_events,
    }
