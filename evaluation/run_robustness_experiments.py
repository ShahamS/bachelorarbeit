import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKMYFLOW_ROOT = REPO_ROOT / "CheckMyFlow_DisCC"
TOKEN_REPLAY_ROOT = REPO_ROOT / "TokenBasedReplay"
DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets"
    / "artificial_log.xes"
    #/ "Hospital_log.xes"
    #/ "Hospital_log.xes"
)

DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "results" / "robustness" / "artificial_inductive_robustness_experiments.csv"


for import_path in (REPO_ROOT, CHECKMYFLOW_ROOT, TOKEN_REPLAY_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from checkmyflow.conformance import OnlineChecker
from checkmyflow.data import EventLogSplitter as CheckMyFlowSplitter
from checkmyflow.model import ModelBuilder

from TokenBasedReplay.tokenbasedreplay.data.splitter import EventLogSplitter as TokenReplaySplitter
from TokenBasedReplay.tokenbasedreplay.data.converter import Converter as TokenReplayConverter
from TokenBasedReplay.tokenbasedreplay.evaluation.metrics import (
    _activity_coverage_metrics as token_activity_coverage_metrics,
    discover_model,
    _model_metrics as token_model_metrics,
    _network_metrics as token_network_metrics,
    _participant_metrics,
    _quality_metrics as token_quality_metrics,
    _result_metrics as token_result_metrics,
    _run_metadata as token_run_metadata,
    _timing_metrics as token_timing_metrics,
)
from TokenBasedReplay.tokenbasedreplay.model.distributed.assignment import ParticipantAssignment
from TokenBasedReplay.tokenbasedreplay.model.distributed.distributed_replayer import (
    DistributedTokenReplayer,
)
from TokenBasedReplay.tokenbasedreplay.model.distributed.network import NetworkSimulator


@dataclass
class ManipulationStats:
    manipulated_events: int = 0
    inserted_events: int = 0
    deleted_events: int = 0
    manipulated_traces: int = 0
    original_events: int = 0
    resulting_events: int = 0

    @property
    def manipulated_events_pct(self):
        if self.original_events == 0:
            return 0.0
        return self.manipulated_events / self.original_events

    @property
    def manipulated_traces_pct(self):
        if self.total_traces == 0:
            return 0.0
        return self.manipulated_traces / self.total_traces

    total_traces: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vergleicht Robustheit von CheckMyFlow und verteiltem Token-Based Replay.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Pfad zum XES-Datensatz.",
    )
    parser.add_argument(
        "--location-key",
        default="org:group",
        help="Event-Attribut fuer die dezentrale Node-/Participant-Zuordnung.",
    )
    parser.add_argument(
        "--training-splits",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8],
        help="Trainingsanteile auf Trace-Ebene.",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=[0.2],
        help="Manipulationswahrscheinlichkeiten pro Event.",
    )
    parser.add_argument(
        "--manipulation-types",
        nargs="+",
        choices=["deletion", "insertion"],
        default=["deletion", "insertion"],
        help="Auszufuehrende Manipulationsarten.",
    )
    parser.add_argument(
        "--max-manipulations-per-trace",
        type=int,
        default=2,
        help="Absolute Obergrenze manipulierter Events pro Trace.",
    )
    parser.add_argument(
        "--max-manipulations-per-trace-pct",
        type=float,
        default=0.25,
        help="Relative Obergrenze manipulierter Events pro Trace.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Seed fuer reproduzierbare Splits und Manipulationen.",
    )
    parser.add_argument(
        "--discovery-algorithm",
        choices=["alpha", "inductive", "heuristics"],
        default="inductive",
        help="Discovery-Algorithmus fuer das Petri-Netz des verteilten TBR.",
    )
    parser.add_argument(
        "--heuristic-threshold",
        type=float,
        default=None,
        help="Dependency Threshold fuer heuristics Discovery.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        choices=["checkmyflow", "distributed_tbr"],
        default=["checkmyflow", "distributed_tbr"],
        help="Zu evaluierende Ansaetze.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Zielpfad fuer die CSV-Ausgabe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    rows = []
    sys.setrecursionlimit(5000)

    for training_split in args.training_splits:
        print(f"Preparing split={training_split}", flush=True)
        cmf_training_log = cmf_test_log = None
        tbr_training_log = tbr_test_log = None

        if "checkmyflow" in args.approaches:
            cmf_training_log, cmf_test_log = load_checkmyflow_split(args, dataset_path, training_split)
            rows.extend(run_checkmyflow_rows(args, dataset_path, training_split, cmf_training_log, cmf_test_log))

        if "distributed_tbr" in args.approaches:
            tbr_training_log, tbr_test_log = load_tbr_split(args, dataset_path, training_split)
            rows.extend(run_tbr_rows(args, dataset_path, training_split, tbr_training_log, tbr_test_log))

    write_rows(output_path, rows)
    print(f"Wrote {output_path}", flush=True)

def load_checkmyflow_split(args, dataset_path, training_split):
    splitter = CheckMyFlowSplitter(
        file_path=str(dataset_path),
        training_split=training_split,
        location_key=args.location_key,
        random_seed=args.random_seed,
    )
    return splitter.split()


def load_tbr_split(args, dataset_path, training_split):
    splitter = TokenReplaySplitter(
        file_path=str(dataset_path),
        training_split=training_split,
        location_key=args.location_key,
        random_seed=args.random_seed,
    )
    return splitter.split()


def run_checkmyflow_rows(args, dataset_path, training_split, training_log, test_log):
    print(f"Training CheckMyFlow split={training_split}", flush=True)
    training_start = time.perf_counter()
    model = ModelBuilder().build(training_log)
    training_time_s = time.perf_counter() - training_start
    fitness_result = OnlineChecker(model).check(training_log)

    baseline_metrics = checkmyflow_metrics(model, test_log, fitness_result, training_time_s)
    rows = [
        _row(
            approach="checkmyflow",
            dataset_path=dataset_path,
            args=args,
            training_split=training_split,
            scenario="baseline",
            manipulation_type="none",
            manipulation_probability=0.0,
            stats=empty_stats(test_log),
            baseline_score=baseline_metrics["generalization"],
            metrics=baseline_metrics,
        )
    ]

    for manipulation_type in args.manipulation_types:
        for noise_level in args.noise_levels:
            seed = scenario_seed(args.random_seed, training_split, manipulation_type, noise_level)
            manipulated_log, stats = manipulate_event_log(
                test_log,
                manipulation_type=manipulation_type,
                probability=noise_level,
                max_per_trace=args.max_manipulations_per_trace,
                max_per_trace_pct=args.max_manipulations_per_trace_pct,
                random_seed=seed,
            )
            metrics = checkmyflow_metrics(model, manipulated_log, fitness_result, training_time_s)
            rows.append(
                _row(
                    approach="checkmyflow",
                    dataset_path=dataset_path,
                    args=args,
                    training_split=training_split,
                    scenario="manipulated",
                    manipulation_type=manipulation_type,
                    manipulation_probability=noise_level,
                    stats=stats,
                    baseline_score=baseline_metrics["generalization"],
                    metrics=metrics,
                )
            )

    return rows


def checkmyflow_metrics(model, event_log, fitness_result, training_time_s):
    checking_start = time.perf_counter()
    result = OnlineChecker(model).check(event_log)
    checking_time_s = time.perf_counter() - checking_start
    total_events = result.total_events
    mismatches = result.total_mismatches
    matches = result.total_matches

    return {
        "total_events": total_events,
        "matches": matches,
        "mismatches": mismatches,
        "generalization": result.generalization,
        "exact_pct": 0.0 if total_events == 0 else matches / total_events,
        "unknown_node": result.reason_count("unknown_node"),
        "invalid_predecessor": result.reason_count("invalid_predecessor"),
        "invalid_end": result.reason_count("invalid_end"),
        "fitness": fitness_result.generalization,
        "fitness_generalization_gap": fitness_result.generalization - result.generalization,
        "route_calls": result.route_calls,
        "remote_calls": result.remote_calls,
        "local_calls": result.local_calls,
        "requests_per_event": 0.0 if total_events == 0 else result.remote_calls / total_events,
        "nodes": len(model.nodes),
        "activities": sum(len(node.activities) for node in model.nodes.values()),
        "matrix_entries": sum(
            len(predecessors)
            for node in model.nodes.values()
            for predecessors in node.footprint_matrix.allowed_predecessors.values()
        ),
        "training_time_s": training_time_s,
        "checking_time_s": checking_time_s,
        "elapsed_s": training_time_s + checking_time_s,
        "avg_event_time_ms": 0.0 if total_events == 0 else (checking_time_s / total_events) * 1000,
    }


def run_tbr_rows(args, dataset_path, training_split, training_log, test_log):
    print(f"Training distributed TBR split={training_split}", flush=True)
    converter = TokenReplayConverter()
    discovery_start = time.perf_counter()
    discovered_model = discover_model(
        converter.from_event_log(training_log),
        args.discovery_algorithm,
        args.heuristic_threshold,
    )
    discovery_time_s = time.perf_counter() - discovery_start
    fitness_result = replay_tbr(discovered_model, training_log, training_log)

    baseline_metrics = tbr_metrics(
        args,
        discovered_model,
        training_log,
        test_log,
        fitness_result,
        discovery_time_s,
    )
    rows = [
        _row(
            approach="distributed_tbr",
            dataset_path=dataset_path,
            args=args,
            training_split=training_split,
            scenario="baseline",
            manipulation_type="none",
            manipulation_probability=0.0,
            stats=empty_stats(test_log),
            baseline_score=baseline_metrics["generalization"],
            metrics=baseline_metrics,
        )
    ]

    for manipulation_type in args.manipulation_types:
        for noise_level in args.noise_levels:
            seed = scenario_seed(args.random_seed, training_split, manipulation_type, noise_level)
            manipulated_log, stats = manipulate_event_log(
                test_log,
                manipulation_type=manipulation_type,
                probability=noise_level,
                max_per_trace=args.max_manipulations_per_trace,
                max_per_trace_pct=args.max_manipulations_per_trace_pct,
                random_seed=seed,
            )
            metrics = tbr_metrics(
                args,
                discovered_model,
                training_log,
                manipulated_log,
                fitness_result,
                discovery_time_s,
            )
            rows.append(
                _row(
                    approach="distributed_tbr",
                    dataset_path=dataset_path,
                    args=args,
                    training_split=training_split,
                    scenario="manipulated",
                    manipulation_type=manipulation_type,
                    manipulation_probability=noise_level,
                    stats=stats,
                    baseline_score=baseline_metrics["generalization"],
                    metrics=metrics,
                )
            )

    return rows


def tbr_metrics(args, discovered_model, training_log, event_log, fitness_result, discovery_time_s):
    replay_start = time.perf_counter()
    replay_result = replay_tbr(discovered_model, training_log, event_log)
    replay_time_s = time.perf_counter() - replay_start
    participants, _, _ = ParticipantAssignment(discovered_model, training_log).build()

    metrics = {}
    metrics.update(token_run_metadata(discovered_model, args.discovery_algorithm, "distributed"))
    metrics.update(token_model_metrics(discovered_model))
    metrics.update(token_activity_coverage_metrics(training_log, discovered_model))
    metrics.update(_participant_metrics(participants))
    metrics.update(token_result_metrics(replay_result))
    metrics.update(token_quality_metrics(fitness_result, replay_result))
    metrics.update(token_network_metrics(replay_result))
    metrics.update(token_timing_metrics(discovery_time_s, replay_time_s, replay_result))
    for key in ("sum_step_loss", "avg_step_loss", "max_step_loss"):
        metrics.pop(key, None)
    return metrics


def replay_tbr(discovered_model, training_log, event_log):
    participants, activity_mapping, place_mapping = ParticipantAssignment(
        discovered_model,
        training_log,
    ).build()
    network = NetworkSimulator(participants, activity_mapping, place_mapping)
    return DistributedTokenReplayer(discovered_model, participants, network).replay_log(event_log)


def manipulate_event_log(
    event_log,
    manipulation_type,
    probability,
    max_per_trace,
    max_per_trace_pct,
    random_seed,
):
    rng = random.Random(random_seed)
    manipulated_log = event_log.__class__()
    templates = event_templates(event_log)
    stats = ManipulationStats(
        original_events=event_count(event_log),
        total_traces=len(event_log),
    )

    for case_id, events in event_log.iter_traces():
        new_events = [clone_event(event) for event in events]
        max_changes = max_changes_for_trace(len(new_events), max_per_trace, max_per_trace_pct)

        if max_changes > 0 and manipulation_type == "deletion":
            changed = delete_events(new_events, probability, max_changes, rng)
            stats.deleted_events += changed
        elif max_changes > 0 and manipulation_type == "insertion":
            changed = insert_events(new_events, case_id, templates, probability, max_changes, rng)
            stats.inserted_events += changed
        else:
            changed = 0

        if changed:
            stats.manipulated_traces += 1
            stats.manipulated_events += changed

        for event in new_events:
            manipulated_log.add_event(event)

    stats.resulting_events = event_count(manipulated_log)
    return manipulated_log, stats


def delete_events(events, probability, max_changes, rng):
    if len(events) <= 1:
        return 0
    candidate_indexes = [index for index in range(len(events)) if rng.random() < probability]
    rng.shuffle(candidate_indexes)
    indexes_to_delete = set(candidate_indexes[:max_changes])
    if not indexes_to_delete:
        return 0
    events[:] = [event for index, event in enumerate(events) if index not in indexes_to_delete]
    return len(indexes_to_delete)


def insert_events(events, case_id, templates, probability, max_changes, rng):
    if not templates:
        return 0
    candidate_positions = [index for index in range(len(events) + 1) if rng.random() < probability]
    rng.shuffle(candidate_positions)
    positions = sorted(candidate_positions[:max_changes], reverse=True)
    for position in positions:
        template = rng.choice(templates)
        timestamp = timestamp_for_insert(events, position)
        events.insert(position, new_event(template, case_id, timestamp))
    return len(positions)


def timestamp_for_insert(events, position):
    if not events:
        return None
    if position <= 0:
        return events[0].time
    return events[position - 1].time


def event_templates(event_log):
    templates = []
    seen = set()
    for _, events in event_log.iter_traces():
        for event in events:
            key = (event.activity, event.location)
            if key not in seen:
                seen.add(key)
                templates.append(event)
    return templates


def clone_event(event):
    return event.__class__(
        case_id=event.case_id,
        activity=event.activity,
        location=event.location,
        time=event.time,
    )


def new_event(template, case_id, timestamp):
    return template.__class__(
        case_id=case_id,
        activity=template.activity,
        location=template.location,
        time=timestamp,
    )


def max_changes_for_trace(trace_length, max_per_trace, max_per_trace_pct):
    if trace_length == 0:
        return 0
    pct_limit = max(1, int(trace_length * max_per_trace_pct))
    return max(0, min(max_per_trace, pct_limit))


def empty_stats(event_log):
    return ManipulationStats(
        original_events=event_count(event_log),
        resulting_events=event_count(event_log),
        total_traces=len(event_log),
    )


def _row(
    approach,
    dataset_path,
    args,
    training_split,
    scenario,
    manipulation_type,
    manipulation_probability,
    stats,
    baseline_score,
    metrics,
):
    manipulated_score = metrics.get("generalization", 0.0)
    row = {
        "approach": approach,
        "dataset": dataset_path.name,
        "training_split": training_split,
        "random_seed": args.random_seed,
        "location_key": args.location_key,
        "scenario": scenario,
        "manipulation_type": manipulation_type,
        "manipulation_probability": manipulation_probability,
        "max_manipulations_per_trace": args.max_manipulations_per_trace,
        "max_manipulations_per_trace_pct": args.max_manipulations_per_trace_pct,
        "baseline_score": baseline_score,
        "manipulated_score": manipulated_score,
        "score_drop": baseline_score - manipulated_score,
        "manipulated_events": stats.manipulated_events,
        "inserted_events": stats.inserted_events,
        "deleted_events": stats.deleted_events,
        "original_events": stats.original_events,
        "resulting_events": stats.resulting_events,
        "manipulated_events_pct": stats.manipulated_events_pct,
        "manipulated_traces": stats.manipulated_traces,
        "total_traces": stats.total_traces,
        "manipulated_traces_pct": stats.manipulated_traces_pct,
    }
    row.update(metrics)
    return row


def scenario_seed(base_seed, training_split, manipulation_type, noise_level):
    value = f"{base_seed}:{training_split}:{manipulation_type}:{noise_level}"
    return sum((index + 1) * ord(char) for index, char in enumerate(value))


def event_count(event_log):
    return sum(len(trace) for _, trace in event_log.iter_traces())


def write_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
