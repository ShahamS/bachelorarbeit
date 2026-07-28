from datetime import datetime
from unittest import TestCase

from checkmyflow.data.event import Event
from checkmyflow.data.event_log import EventLog
from checkmyflow.evaluation import evaluate_checkmyflow


def _event(case_id, activity, node_id):
    return Event(case_id, activity, node_id, datetime(2026, 1, 1))


class EvaluationMetricsTest(TestCase):
    def test_evaluation_exports_distributed_alignment_compatible_metrics(self):
        training_log = EventLog(
            {
                "case_train": [
                    _event("case_train", "A", "node-1"),
                    _event("case_train", "B", "node-2"),
                ]
            }
        )
        test_log = EventLog(
            {
                "case_match": [
                    _event("case_match", "A", "node-1"),
                    _event("case_match", "B", "node-2"),
                ],
                "case_mismatch": [
                    _event("case_mismatch", "A", "node-1"),
                    _event("case_mismatch", "C", "node-2"),
                ],
            }
        )

        metrics = evaluate_checkmyflow(training_log, test_log).to_dict()

        self.assertEqual(metrics["total_events"], 4)
        self.assertEqual(metrics["matches"], 3)
        self.assertEqual(metrics["mismatches"], 1)
        self.assertEqual(metrics["sum_step_loss"], 1)
        self.assertEqual(metrics["max_step_loss"], 1)
        self.assertEqual(metrics["exact_count"], 3)
        self.assertEqual(metrics["total_route"], 4)
        self.assertEqual(metrics["total_remote"], 0)
        self.assertEqual(metrics["total_queried"], 4)
        self.assertEqual(metrics["total_compute"], 4)
        self.assertEqual(metrics["nodes"], 2)
        self.assertEqual(metrics["matrix_entries"], 2)
        self.assertEqual(metrics["final_states"], 2)

    def test_evaluation_counts_unknown_nodes_separately(self):
        training_log = EventLog(
            {
                "case_train": [
                    _event("case_train", "A", "node-1"),
                ]
            }
        )
        test_log = EventLog(
            {
                "case_unknown_node": [
                    _event("case_unknown_node", "A", "node-9"),
                ],
            }
        )

        metrics = evaluate_checkmyflow(training_log, test_log).to_dict()

        self.assertEqual(metrics["mismatches"], 1)
        self.assertEqual(metrics["unknown_node"], 1)
        self.assertEqual(metrics["invalid_predecessor"], 0)
