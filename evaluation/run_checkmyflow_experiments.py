'''Führt reproduzierbare CheckMyFlow-Experimente aus.
Das Skript erzeugt eine gemeinsame CSV-Datei für mehrere Trainings-Splits.
Diese Tabelle ist als Gegenstück zu den CSV-Ausgaben der
`distributed-alignments`-Evaluation gedacht.
'''


import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKMYFLOW_ROOT = REPO_ROOT / "CheckMyFlow_DisCC"
DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets"
    / "Real-life event logs - Hospital log_1_all"
    / "Hospital_log.xes"
    / "Hospital_log.xes"
)
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "results" / "checkmyflow" / "hospital_training_splits.csv"


if str(CHECKMYFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(CHECKMYFLOW_ROOT))

from checkmyflow.data import EventLogSplitter
from checkmyflow.evaluation import evaluate_checkmyflow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuehrt CheckMyFlow fuer mehrere Trainings-Splits aus.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Pfad zum XES-Datensatz.",
    )
    parser.add_argument(
        "--location-key",
        default=None,
        help="Event-Attribut, das als Node verwendet wird.",
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
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Zielpfad fuer die gemeinsame CSV-Ausgabe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    rows = []
    for training_split in args.training_splits:
        print(f"Running CheckMyFlow split={training_split}")
        # Erzeuge EventLogs mit interner Struktur und splitte sie in Trainings- und Testdaten
        splitter = EventLogSplitter(
            file_path=str(dataset_path),
            training_split=training_split,
            location_key=args.location_key,
            random_seed=args.random_seed,
        )
        training_log, test_log = splitter.split()
        # Führe CheckMyFlow aus und sammle die Ergebnisse
        summary = evaluate_checkmyflow(training_log, test_log)

        row = {
            "approach": "checkmyflow",
            "dataset": dataset_path.name,
            "training_split": training_split,
            "random_seed": args.random_seed,
            "location_key": args.location_key,
            "training_traces": len(training_log),
            "test_traces": len(test_log),
        }
        row.update(summary.to_dict())
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
