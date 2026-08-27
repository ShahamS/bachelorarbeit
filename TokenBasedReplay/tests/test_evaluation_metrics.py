from unittest import TestCase
from unittest.mock import patch

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from tokenbasedreplay.data.event import Event
from tokenbasedreplay.data.event_log import EventLog
from tokenbasedreplay.evaluation import (
    evaluate_central_token_replay,
    evaluate_distributed_token_replay,
)
from tokenbasedreplay.model.petrinet.discovered_model import DiscoveredPetriNet


def _two_transition_model():
    net = PetriNet("two-step")
    start = PetriNet.Place("start")
    middle = PetriNet.Place("middle")
    end = PetriNet.Place("end")
    transition_a = PetriNet.Transition("t_a", "A")
    transition_b = PetriNet.Transition("t_b", "B")

    net.places.update({start, middle, end})
    net.transitions.update({transition_a, transition_b})
    petri_utils.add_arc_from_to(start, transition_a, net)
    petri_utils.add_arc_from_to(transition_a, middle, net)
    petri_utils.add_arc_from_to(middle, transition_b, net)
    petri_utils.add_arc_from_to(transition_b, end, net)

    initial_marking = Marking()
    initial_marking[start] = 1
    final_marking = Marking()
    final_marking[end] = 1

    return DiscoveredPetriNet("manual", {"source": "test"}, net, initial_marking, final_marking)


class DistributedTokenReplayEvaluationTest(TestCase):
    def test_evaluate_central_token_replay_keeps_network_columns_at_zero(self):
        training_log = EventLog()
        training_log.add_event(Event("training", "A", "node-1", 1))
        training_log.add_event(Event("training", "B", "node-2", 2))
        test_log = EventLog()
        test_log.add_event(Event("case-1", "A", "node-1", 3))
        test_log.add_event(Event("case-1", "B", "node-2", 4))

        with patch(
            "tokenbasedreplay.evaluation.metrics._discover_model",
            return_value=_two_transition_model(),
        ):
            summary = evaluate_central_token_replay(training_log, test_log)

        metrics = summary.to_dict()

        self.assertEqual(metrics["strategy"], "central")
        self.assertEqual(metrics["route_calls"], 0)
        self.assertEqual(metrics["remote_calls"], 0)
        self.assertEqual(metrics["local_calls"], 0)

    def test_evaluate_distributed_token_replay_returns_csv_ready_metrics(self):
        training_log = EventLog()
        training_log.add_event(Event("training", "A", "node-1", 1))
        training_log.add_event(Event("training", "B", "node-2", 2))
        test_log = EventLog()
        test_log.add_event(Event("case-1", "A", "node-1", 3))
        test_log.add_event(Event("case-1", "B", "node-2", 4))

        with patch(
            "tokenbasedreplay.evaluation.metrics._discover_model",
            return_value=_two_transition_model(),
        ):
            summary = evaluate_distributed_token_replay(training_log, test_log)

        metrics = summary.to_dict()

        self.assertEqual(metrics["method"], "token_replay")
        self.assertEqual(metrics["strategy"], "distributed")
        self.assertEqual(metrics["participants"], 2)
        self.assertEqual(metrics["total_events"], 2)
        self.assertEqual(metrics["consumed_tokens"], 2)
        self.assertEqual(metrics["produced_tokens"], 2)
        self.assertEqual(metrics["missing_tokens"], 0)
        self.assertEqual(metrics["remaining_tokens"], 0)
        self.assertEqual(metrics["fitness"], 1.0)
        self.assertEqual(metrics["route_calls"], 2)
        self.assertEqual(metrics["remote_calls"], 1)
        self.assertEqual(metrics["local_token_checks"], 1)
        self.assertEqual(metrics["remote_token_checks"], 1)
