from datetime import datetime
from unittest import TestCase

from checkmyflow.conformance import OnlineChecker
from checkmyflow.data.event import Event
from checkmyflow.data.event_log import EventLog
from checkmyflow.model import ModelBuilder


def _event(case_id, activity, node_id):
    return Event(case_id, activity, node_id, datetime(2026, 1, 1))


class TrainingAndConformanceTest(TestCase):
    def test_training_learns_start_and_direct_successor_relations(self):
        training_log = EventLog(
            {
                "case_train": [
                    _event("case_train", "A", "node-1"),
                    _event("case_train", "B", "node-2"),
                ]
            }
        )

        model = ModelBuilder().build(training_log)

        self.assertTrue(
            model.allows_event_relation("__START__", _event("case_test", "A", "node-1"))
        )
        self.assertTrue(
            model.allows_event_relation("A", _event("case_test", "B", "node-2"))
        )

    def test_online_checker_counts_matches_and_mismatches_without_learning_from_test_log(self):
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
                "case_wrong_activity": [
                    _event("case_wrong_activity", "A", "node-1"),
                    _event("case_wrong_activity", "C", "node-2"),
                ],
                "case_wrong_node": [
                    _event("case_wrong_node", "A", "node-1"),
                    _event("case_wrong_node", "B", "node-9"),
                ],
            }
        )

        model = ModelBuilder().build(training_log)
        result = OnlineChecker(model).check(test_log)

        self.assertEqual(result.total_events, 6)
        self.assertEqual(result.total_matches, 4)
        self.assertEqual(result.total_mismatches, 2)
        self.assertIsNone(
            model.get_existing_node_for_event(_event("case_test", "B", "node-9"))
        )
