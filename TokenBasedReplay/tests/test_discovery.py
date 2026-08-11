from datetime import datetime
from unittest import TestCase

from tokenbasedreplay.data.converter import Converter
from tokenbasedreplay.data.event import Event
from tokenbasedreplay.data.event_log import EventLog
from tokenbasedreplay.model.petrinet.discoverer import DiscoveryRunner

'''
Testet die DiscoveryRunner-Klasse, die Petri-Netze aus Event-Logs entdeckt mit
verschiedenen Algorithmen (Alpha, Heuristics, Inductive).
Die Tests prüfen, ob die entdeckten Modelle die erwarteten Eigenschaften haben,
wie z.B. die Anzahl der Plätze, Transitionen und Kanten.
'''

def _event(case_id, activity, node_id, day):
    return Event(case_id, activity, node_id, datetime(2026, 1, day))


def _simple_event_log():
    return EventLog(
        {
            "case_1": [
                _event("case_1", "A", "node-1", 1),
                _event("case_1", "B", "node-2", 2),
            ],
            "case_2": [
                _event("case_2", "A", "node-1", 3),
                _event("case_2", "B", "node-2", 4),
            ],
        }
    )


class DiscoveryRunnerTest(TestCase):
    def setUp(self):
        self.pm4py_log = Converter().from_event_log(_simple_event_log())
        self.runner = DiscoveryRunner()

    def test_alpha_discovery_returns_discovered_petri_net(self):
        model = self.runner.discover_alpha(self.pm4py_log)

        self.assertEqual(model.algorithm, "alpha")
        self.assertEqual(model.parameters, {})
        self.assertGreater(model.places_count(), 0)
        self.assertGreater(model.transitions_count(), 0)
        self.assertGreater(model.arcs_count(), 0)
        self.assertEqual(model.labeled_transitions_count(), 2)

    def test_inductive_discovery_returns_discovered_petri_net(self):
        model = self.runner.discover_inductive(self.pm4py_log)

        self.assertEqual(model.algorithm, "inductive")
        self.assertEqual(model.parameters, {})
        self.assertGreater(model.places_count(), 0)
        self.assertGreater(model.transitions_count(), 0)
        self.assertGreater(model.arcs_count(), 0)
        self.assertEqual(model.labeled_transitions_count(), 2)

    def test_heuristics_discovery_keeps_dependency_threshold(self):
        model = self.runner.discover_heuristics(self.pm4py_log, 0.8)

        self.assertEqual(model.algorithm, "heuristics")
        self.assertEqual(model.parameters, {"dependency_threshold": 0.8})
        self.assertGreater(model.places_count(), 0)
        self.assertGreater(model.transitions_count(), 0)
        self.assertGreater(model.arcs_count(), 0)
        self.assertEqual(model.labeled_transitions_count(), 2)

    def test_discover_all_includes_base_and_threshold_variants(self):
        models = self.runner.discover_all(self.pm4py_log, heuristic_thresholds=[0.8, 0.9])

        self.assertEqual([model.algorithm for model in models], [
            "alpha",
            "inductive",
            "heuristics",
            "heuristics",
        ])
        self.assertEqual(models[2].parameters, {"dependency_threshold": 0.8})
        self.assertEqual(models[3].parameters, {"dependency_threshold": 0.9})
