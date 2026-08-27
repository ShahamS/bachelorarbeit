"""Evaluation API fuer zentrale und dezentrale Token-Based-Replay-Laeufe."""

import csv
import json
import time
from pathlib import Path

from tokenbasedreplay.data.converter import Converter
from tokenbasedreplay.model.central.central_replayer import CentralTokenReplayer
from tokenbasedreplay.model.distributed.assignment import ParticipantAssignment
from tokenbasedreplay.model.distributed.distributed_replayer import DistributedTokenReplayer
from tokenbasedreplay.model.distributed.network import NetworkSimulator
from tokenbasedreplay.model.petrinet.discoverer import DiscoveryRunner


class TokenReplayEvaluationSummary:
    """Sammelt die Kennzahlen eines Token-Replay-Laufs."""

    def __init__(self, metrics, replay_result=None, discovered_model=None):
        self.metrics = metrics
        self.replay_result = replay_result
        self.discovered_model = discovered_model

    def to_dict(self):
        return dict(self.metrics)

    def write_csv(self, file_path):
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(self.metrics.keys()))
            writer.writeheader()
            writer.writerow(self.metrics)

    def write_json(self, file_path):
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(self.metrics, json_file, indent=2)


def evaluate_central_token_replay(
    training_log,
    test_log,
    discovery_algorithm="inductive",
    heuristic_threshold=None,
):
    """Entdeckt ein Petri-Netz und evaluiert zentrales Token-Based Replay."""

    converter = Converter()
    discovery_start = time.perf_counter()
    pm4py_training_log = converter.from_event_log(training_log)
    discovered_model = _discover_model(
        pm4py_training_log,
        discovery_algorithm,
        heuristic_threshold,
    )
    discovery_time_s = time.perf_counter() - discovery_start

    replay_start = time.perf_counter()
    replay_result = CentralTokenReplayer(discovered_model).replay_log(test_log)
    replay_time_s = time.perf_counter() - replay_start

    metrics = {}
    metrics.update(_run_metadata(discovered_model, discovery_algorithm))
    metrics.update(_log_metrics(training_log, test_log))
    metrics.update(_model_metrics(discovered_model))
    metrics.update(_result_metrics(replay_result))
    metrics.update(_network_metrics(replay_result))
    metrics.update(_timing_metrics(discovery_time_s, replay_time_s, replay_result))

    return TokenReplayEvaluationSummary(
        metrics=metrics,
        replay_result=replay_result,
        discovered_model=discovered_model,
    )


def evaluate_distributed_token_replay(
    training_log,
    test_log,
    discovery_algorithm="alpha",
    heuristic_threshold=None,
):
    # Entdeckt ein Petri-Netz und evaluiert dezentrales Token-Based Replay

    converter = Converter()
    discovery_start = time.perf_counter()
    pm4py_training_log = converter.from_event_log(training_log)
    # Trainiere Model in Form eines Petri-Netz
    discovered_model = _discover_model(
        pm4py_training_log,
        discovery_algorithm,
        heuristic_threshold,
    )
    # Ordne Participants Bereichen des Petri-Netzes zu
    participants, activity_mapping, place_mapping = ParticipantAssignment(
        discovered_model,
        training_log,
    ).build()
    discovery_time_s = time.perf_counter() - discovery_start

    replay_start = time.perf_counter()
    network = NetworkSimulator(participants, activity_mapping, place_mapping)
    # Starte dezentrales Token-Based Replay
    replay_result = DistributedTokenReplayer(
        discovered_model,
        participants,
        network,
    ).replay_log(test_log)
    replay_time_s = time.perf_counter() - replay_start

    metrics = {}
    metrics.update(_run_metadata(discovered_model, discovery_algorithm, "distributed"))
    metrics.update(_log_metrics(training_log, test_log))
    metrics.update(_model_metrics(discovered_model))
    metrics.update(_participant_metrics(participants))
    metrics.update(_result_metrics(replay_result))
    metrics.update(_network_metrics(replay_result))
    metrics.update(_timing_metrics(discovery_time_s, replay_time_s, replay_result))

    return TokenReplayEvaluationSummary(
        metrics=metrics,
        replay_result=replay_result,
        discovered_model=discovered_model,
    )


def _discover_model(pm4py_training_log, discovery_algorithm, heuristic_threshold):
    discovery = DiscoveryRunner()

    if discovery_algorithm == "alpha":
        return discovery.discover_alpha(pm4py_training_log)
    if discovery_algorithm == "inductive":
        return discovery.discover_inductive(pm4py_training_log)
    if discovery_algorithm == "heuristics":
        threshold = 0.8 if heuristic_threshold is None else heuristic_threshold
        return discovery.discover_heuristics(pm4py_training_log, threshold)

    raise ValueError(f"Unbekannter Discovery-Algorithmus: {discovery_algorithm}")


def _run_metadata(discovered_model, requested_algorithm, strategy="central"):
    return {
        "method": "token_replay",
        "strategy": strategy,
        "discovery_algorithm": discovered_model.algorithm,
        "requested_discovery_algorithm": requested_algorithm,
        "discovery_parameters": json.dumps(discovered_model.parameters, sort_keys=True),
    }


def _log_metrics(training_log, test_log):
    return {
        "training_traces": len(training_log),
        "test_traces": len(test_log),
    }


def _model_metrics(discovered_model):
    return {
        "places": discovered_model.places_count(),
        "transitions": discovered_model.transitions_count(),
        "arcs": discovered_model.arcs_count(),
        "labeled_transitions": discovered_model.labeled_transitions_count(),
        "silent_transitions": discovered_model.silent_transitions_count(),
    }


def _participant_metrics(participants):
    participant_count = len(participants)
    place_counts = [len(participant.places) for participant in participants.values()]
    transition_counts = [len(participant.transitions) for participant in participants.values()]

    return {
        "participants": participant_count,
        "participant_places_min": min(place_counts, default=0),
        "participant_places_max": max(place_counts, default=0),
        "participant_places_total": sum(place_counts),
        "participant_transitions_min": min(transition_counts, default=0),
        "participant_transitions_max": max(transition_counts, default=0),
        "participant_transitions_total": sum(transition_counts),
    }


def _result_metrics(result):
    total_events = result.total_events
    mismatches = total_events - result.matched_events

    if total_events == 0:
        avg_step_loss = 0.0
        exact_pct = 0.0
    else:
        avg_step_loss = mismatches / total_events
        exact_pct = result.matched_events / total_events

    return {
        "total_events": total_events,
        "matched_events": result.matched_events,
        "mismatches": mismatches,
        "unknown_activities": result.unknown_activities,
        "consumed_tokens": result.total_consumed_tokens,
        "produced_tokens": result.total_produced_tokens,
        "missing_tokens": result.total_missing_tokens,
        "remaining_tokens": result.total_remaining_tokens,
        "fitness": result.fitness,
        "sum_step_loss": mismatches,
        "avg_step_loss": avg_step_loss,
        "max_step_loss": 1 if mismatches else 0,
        "exact_count": result.matched_events,
        "exact_pct": exact_pct,
    }


def _network_metrics(result):
    network_metrics = getattr(result, "network_metrics", {})
    return {
        "route_calls": getattr(result, "route_calls", 0),
        "remote_calls": getattr(result, "remote_calls", 0),
        "local_calls": getattr(result, "local_calls", 0),
        "local_token_checks": network_metrics.get("local_token_checks", 0),
        "remote_token_checks": network_metrics.get("remote_token_checks", 0),
    }


def _timing_metrics(discovery_time_s, replay_time_s, result):
    elapsed_s = discovery_time_s + replay_time_s
    total_events = result.total_events
    avg_event_time_ms = 0.0
    if total_events > 0:
        avg_event_time_ms = (replay_time_s / total_events) * 1000

    return {
        "training_time_s": discovery_time_s,
        "checking_time_s": replay_time_s,
        "elapsed_s": elapsed_s,
        "avg_event_time_ms": avg_event_time_ms,
        "total_compute": total_events,
    }
