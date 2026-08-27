"""CLI-Runner fuer die zentrale Token-Based-Replay-Baseline."""

import argparse
from pathlib import Path

from tokenbasedreplay.data.splitter import EventLogSplitter
from tokenbasedreplay.evaluation import evaluate_central_token_replay


def run_experiment(
    file_path,
    training_split=0.8,
    location_key=None,
    random_seed=1,
    discovery_algorithm="inductive",
    heuristic_threshold=None,
    output_csv=None,
    output_json=None,
):
    splitter = EventLogSplitter(
        file_path=file_path,
        training_split=training_split,
        location_key=location_key,
        random_seed=random_seed,
    )
    training_log, test_log = splitter.split()

    summary = evaluate_central_token_replay(
        training_log=training_log,
        test_log=test_log,
        discovery_algorithm=discovery_algorithm,
        heuristic_threshold=heuristic_threshold,
    )
    metrics = summary.to_dict()
    metrics["dataset"] = Path(file_path).name
    metrics["training_split"] = training_split
    metrics["random_seed"] = random_seed
    metrics["location_key"] = location_key
    summary.metrics = metrics

    if output_csv:
        summary.write_csv(output_csv)
    if output_json:
        summary.write_json(output_json)

    print("Central Token-Based Replay experiment")
    print(f"Log: {Path(file_path)}")
    print(f"Training traces: {len(training_log)}")
    print(f"Test traces: {len(test_log)}")
    print(f"Discovery algorithm: {metrics['discovery_algorithm']}")
    print(f"Checked events: {metrics['total_events']}")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print(f"Missing tokens: {metrics['missing_tokens']}")
    print(f"Remaining tokens: {metrics['remaining_tokens']}")
    print(f"Elapsed seconds: {metrics['elapsed_s']:.6f}")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuehrt die zentrale Token-Based-Replay-Baseline aus.",
    )
    parser.add_argument("file_path", help="Pfad zu einem XES-Eventlog.")
    parser.add_argument(
        "--training-split",
        type=float,
        default=0.8,
        help="Anteil der Traces fuer das Training, default: 0.8.",
    )
    parser.add_argument(
        "--location-key",
        default=None,
        help="Optionales Event-Attribut fuer Location. Fuer zentrale TBR nicht erforderlich.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Seed fuer reproduzierbare Trace-Splits.",
    )
    parser.add_argument(
        "--discovery-algorithm",
        choices=["alpha", "inductive", "heuristics"],
        default="inductive",
        help="PM4Py Discovery-Algorithmus fuer das Petri-Netz.",
    )
    parser.add_argument(
        "--heuristic-threshold",
        type=float,
        default=None,
        help="Dependency Threshold fuer heuristics Discovery.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optionaler Pfad fuer eine einzeilige CSV mit Evaluationsmetriken.",
    )
    parser.add_argument(
        "--output-json",
        help="Optionaler Pfad fuer eine JSON-Datei mit Evaluationsmetriken.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(
        file_path=args.file_path,
        training_split=args.training_split,
        location_key=args.location_key,
        random_seed=args.random_seed,
        discovery_algorithm=args.discovery_algorithm,
        heuristic_threshold=args.heuristic_threshold,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
