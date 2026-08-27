"""Fuehrt zentrale Token-Replay-Experimente fuer mehrere Trainingssplits aus."""

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
    / "Sepsis Cases - Event Log_1_all"
    / "Sepsis Cases - Event Log.xes"
    / "Sepsis Cases - Event Log.xes"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "evaluation"
    / "results"
    / "token_replay"
    / "central_sepsis_training_splits.csv"
)


if str(TOKEN_REPLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKEN_REPLAY_ROOT))

from tokenbasedreplay.data.splitter import EventLogSplitter
from tokenbasedreplay.evaluation import evaluate_central_token_replay


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuehrt zentrale Token-Replay-Experimente fuer mehrere Trainingssplits aus.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Pfad zum XES-Datensatz.",
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
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Zielpfad fuer die gemeinsame CSV-Ausgabe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

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
            f"Running central Token Replay split={training_split} "
            f"training_traces={len(training_log)} test_traces={len(test_log)}",
            flush=True,
        )
        summary = evaluate_central_token_replay(
            training_log=training_log,
            test_log=test_log,
            discovery_algorithm=args.discovery_algorithm,
            heuristic_threshold=args.heuristic_threshold,
        )

        row = {
            "dataset": dataset_path.name,
            "training_split": training_split,
            "random_seed": args.random_seed,
            "location_key": args.location_key,
        }
        row.update(summary.to_dict())
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")


def _event_count(event_log):
    return sum(len(trace) for _, trace in event_log.iter_traces())


if __name__ == "__main__":
    main()
