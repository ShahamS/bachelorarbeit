"""Fuehrt dezentrale Token-Replay-Experimente fuer mehrere Trainingssplits aus."""

import argparse
import csv
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_REPLAY_ROOT = REPO_ROOT / "TokenBasedReplay"
DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets"
    / "Road Traffic Fine Management Process_1_all"
    / "Road_Traffic_Fine_Management_Process.xes"
    / "Road_Traffic_Fine_Management_Process.xes"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "evaluation"
    / "results"
    / "token_replay"
    / "distributed_traffic_training_splits.csv"
)


for import_path in (REPO_ROOT, TOKEN_REPLAY_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from TokenBasedReplay.tokenbasedreplay.data.splitter import EventLogSplitter
from TokenBasedReplay.tokenbasedreplay.evaluation import (
    evaluate_central_token_replay,
    evaluate_distributed_token_replay,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuehrt dezentrale Token-Replay-Experimente fuer mehrere Trainingssplits aus.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Pfad zum XES-Datensatz.",
    )
    parser.add_argument(
        "--location-key",
        required=True,
        help="Event-Attribut fuer die Participant-Zuordnung, z.B. org:group oder org:resource.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Seed fuer reproduzierbare Trace-Splits.",
    )
    parser.add_argument(
        "--training-splits",
        nargs="+",
        type=float,
        default=[0.2, 0.4, 0.6, 0.8],
        help="Liste von Trainingsanteilen, z.B. 0.2 0.4 0.6 0.8.",
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
        "--include-central",
        action="store_true",
        help="Schreibt zusaetzlich zentrale TBR-Zeilen fuer denselben Split.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Zielpfad fuer die gemeinsame CSV-Ausgabe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    sys.setrecursionlimit(5000)

    print(f"Loading dataset {dataset_path}", flush=True)
    load_start = time.perf_counter()
    base_splitter = EventLogSplitter(
        file_path=str(dataset_path),
        training_split=1.0,
        location_key=args.location_key,
        random_seed=args.random_seed,
    )
    base_log = base_splitter.log
    print(
        "Loaded "
        f"{len(base_log)} traces / {_event_count(base_log)} events "
        f"in {time.perf_counter() - load_start:.2f}s",
        flush=True,
    )

    rows = []
    for training_split in args.training_splits:
        print(f"Preparing split={training_split}", flush=True)
        splitter = EventLogSplitter(
            event_log=base_log,
            training_split=training_split,
            location_key=args.location_key,
            random_seed=args.random_seed,
        )
        training_log, test_log = splitter.split()
        print(
            f"Running distributed Token Replay split={training_split} "
            f"training_traces={len(training_log)} test_traces={len(test_log)}",
            flush=True,
        )

        distributed_summary = evaluate_distributed_token_replay(
            training_log=training_log,
            test_log=test_log,
            discovery_algorithm=args.discovery_algorithm,
            heuristic_threshold=args.heuristic_threshold,
        )
        rows.append(
            _result_row(
                dataset_path=dataset_path,
                training_split=training_split,
                random_seed=args.random_seed,
                location_key=args.location_key,
                summary=distributed_summary,
            )
        )

        if args.include_central:
            print(f"Running central Token Replay split={training_split}", flush=True)
            central_summary = evaluate_central_token_replay(
                training_log=training_log,
                test_log=test_log,
                discovery_algorithm=args.discovery_algorithm,
                heuristic_threshold=args.heuristic_threshold,
            )
            rows.append(
                _result_row(
                    dataset_path=dataset_path,
                    training_split=training_split,
                    random_seed=args.random_seed,
                    location_key=args.location_key,
                    summary=central_summary,
                )
            )

    _write_rows(output_path, rows)
    print(f"Wrote {output_path}")


def _result_row(dataset_path, training_split, random_seed, location_key, summary):
    row = {
        "dataset": dataset_path.name,
        "training_split": training_split,
        "random_seed": random_seed,
        "location_key": location_key,
    }
    row.update(summary.to_dict())
    return row


def _write_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _event_count(event_log):
    return sum(len(trace) for _, trace in event_log.iter_traces())


def _fieldnames(rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


if __name__ == "__main__":
    main()
