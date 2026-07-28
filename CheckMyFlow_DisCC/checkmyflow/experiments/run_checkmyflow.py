"""Kleiner Einstiegspunkt fuer CheckMyFlow-Experimente.

Das Skript bildet den geplanten Ablauf ab:

1. XES-Log laden und auf Trace-Ebene in Training/Test splitten.
2. Aus den Trainingsdaten lokale Footprint-Matrizen pro Node lernen.
3. Die Testdaten online gegen das trainierte Modell pruefen.
4. Match-/Mismatch-Zaehler als einfache Ergebnisuebersicht ausgeben.
"""

import argparse
from pathlib import Path

from checkmyflow.data import EventLogSplitter
from checkmyflow.evaluation import evaluate_checkmyflow


def run_experiment(
    file_path,
    training_split=0.8,
    location_key="org:group",
    random_seed=1,
    output_csv=None,
    output_json=None,
):
    """Fuehrt Training und Online Checking fuer ein XES-Log aus."""

    splitter = EventLogSplitter(
        file_path=file_path,
        training_split=training_split,
        location_key=location_key,
        random_seed=random_seed,
    )
    training_log, test_log = splitter.split()

    # Die Evaluation kapselt Training, Online Checking und Messung. Dadurch
    # bleiben Modellzustand, Conformance-Ergebnis und Metrikexport getrennt.
    summary = evaluate_checkmyflow(training_log, test_log)
    metrics = summary.to_dict()

    if output_csv:
        summary.write_csv(output_csv)
    if output_json:
        summary.write_json(output_json)

    print("CheckMyFlow experiment")
    print(f"Log: {Path(file_path)}")
    print(f"Training traces: {len(training_log)}")
    print(f"Test traces: {len(test_log)}")
    print(f"Nodes: {metrics['nodes']}")
    print(f"Checked events: {metrics['total_events']}")
    print(f"Matches: {metrics['matches']}")
    print(f"Mismatches: {metrics['mismatches']}")
    print(f"Fitness: {metrics['fitness']:.4f}")
    print(f"Average step loss: {metrics['avg_step_loss']:.4f}")
    print(f"Exact prefix percentage: {metrics['exact_pct']:.4f}")
    print(f"Route calls: {metrics['total_route']}")
    print(f"Remote calls: {metrics['total_remote']}")
    print(f"Matrix entries: {metrics['matrix_entries']}")
    print(f"Elapsed seconds: {metrics['elapsed_s']:.6f}")


def parse_args():
    """Liest CLI-Parameter fuer reproduzierbare lokale Experimente."""

    parser = argparse.ArgumentParser(
        description="Trainiere CheckMyFlow auf Trainings-Traces und pruefe Test-Traces online.",
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
        default="org:group",
        help="Event-Attribut, das die verteilte Node beschreibt.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Seed fuer den reproduzierbaren Trace-Split.",
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
    """CLI-Wrapper."""

    args = parse_args()
    run_experiment(
        file_path=args.file_path,
        training_split=args.training_split,
        location_key=args.location_key,
        random_seed=args.random_seed,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
