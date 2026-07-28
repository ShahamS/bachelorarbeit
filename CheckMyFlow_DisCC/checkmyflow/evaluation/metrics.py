"""Gemeinsame Evaluationsmetriken fuer CheckMyFlow.

Die Namen orientieren sich bewusst an der `distributed-alignments`-Evaluation.
So koennen beide Ansaetze spaeter in aehnlichen CSV-Dateien verglichen werden,
auch wenn die interne Bedeutung leicht unterschiedlich ist:

- distributed-alignments vergleicht Alignment-Kosten.
- CheckMyFlow prueft lokale Footprint-Regeln.

Fuer CheckMyFlow wird ein Mismatch deshalb als ein Schritt Verlust gezaehlt.
Ein Match hat Verlust 0. Dadurch entsprechen `sum_step_loss`,
`avg_step_loss`, `exact_count` und `exact_pct` der Qualitaet des Online Checks.
"""

import csv
import json
import time

from checkmyflow.conformance import OnlineChecker
from checkmyflow.model import ModelBuilder


class EvaluationSummary:
    """Sammelt alle Kennzahlen eines CheckMyFlow-Laufs."""

    def __init__(self, metrics):
        self.metrics = metrics

    def to_dict(self):
        """Gibt eine Kopie der Kennzahlen fuer CSV/JSON/Tests zurueck."""

        return dict(self.metrics)

    def write_csv(self, file_path):
        """Schreibt eine einzeilige CSV-Datei mit stabiler Spaltenreihenfolge."""

        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(self.metrics.keys()))
            writer.writeheader()
            writer.writerow(self.metrics)

    def write_json(self, file_path):
        """Schreibt die Kennzahlen als gut lesbare JSON-Datei."""

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(self.metrics, json_file, indent=2)


def evaluate_checkmyflow(training_log, test_log):
    """Trainiert CheckMyFlow und misst vergleichbare Evaluationskennzahlen."""

    training_start = time.perf_counter()
    # Baue Model in Form einer Footprintmatrix mit Hilfe des Traininglogs
    model = ModelBuilder().build(training_log)
    training_time_s = time.perf_counter() - training_start

    checking_start = time.perf_counter()
    # Führe Online-Conformance Checking auf dem Testlog mit dem trainierten Model durch und sammle die Ergebnisse
    result = OnlineChecker(model).check(test_log)
    checking_time_s = time.perf_counter() - checking_start

    # Berechne Metriken für Modell, Ergebnis, Netzwerk und Laufzeit
    model_metrics = _model_metrics(model)
    result_metrics = _result_metrics(result)
    network_metrics = _network_metrics(result)
    timing_metrics = _timing_metrics(training_time_s, checking_time_s, result)

    metrics = {}
    metrics.update(result_metrics)
    metrics.update(network_metrics)
    metrics.update(model_metrics)
    metrics.update(timing_metrics)

    return EvaluationSummary(metrics)


def _result_metrics(result):
    '''Berechnet Qualitätsmetriken analog zu Alignment-Step-Ergebnissen'''

    total_events = result.total_events
    mismatches = result.total_mismatches
    matches = result.total_matches

    if total_events == 0:
        avg_step_loss = 0.0
        exact_pct = 0.0
    else:
        avg_step_loss = mismatches / total_events
        exact_pct = matches / total_events

    return {
        "total_events": total_events,
        "matches": matches,
        "mismatches": mismatches,
        "fitness": result.fitness,
        "sum_step_loss": mismatches,
        "avg_step_loss": avg_step_loss,
        "max_step_loss": 1 if mismatches else 0,
        "exact_count": matches,
        "exact_pct": exact_pct,
        "unknown_node": result.reason_count("unknown_node"),
        "invalid_predecessor": result.reason_count("invalid_predecessor"),
    }


def _network_metrics(result):
    '''Schätzt die logischen Kommunikationskosten des verteilten Checks.
    Unsere aktuelle CheckMyFlow-Version prüft jede Relation lokal in der Matrix
    der Event-Node. Darum gibt es pro Event einen Routing-Schritt und eine lokale
    Matrixabfrage, aber noch keine Remote-Anfrage zwischen Nodes.
    '''

    total_events = result.total_events
    return {
        "total_route": total_events,
        "total_remote": 0,
        "total_queried": total_events,
        "requests_per_event": 0.0 if total_events == 0 else total_events / total_events,
    }


def _model_metrics(model):
    """Misst die Größe der lokalen Footprint-Matrizen."""

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
        "final_states": relation_count,
        "max_node_matrix_entries": max_node_relations,
    }


def _timing_metrics(training_time_s, checking_time_s, result):
    """Berechnet Laufzeitmetriken fuer Training und Online Checking."""

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
