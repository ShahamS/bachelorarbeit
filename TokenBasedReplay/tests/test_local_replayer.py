from unittest import TestCase

from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from tokenbasedreplay.model.participant.local_replayer import LocalReplayer
from tokenbasedreplay.model.participant.participant import Participant
from tokenbasedreplay.model.distributed.network import NetworkSimulator
from tokenbasedreplay.model.petrinet.discovered_model import DiscoveredPetriNet


def _single_transition_model():
    net = PetriNet("mini")
    start = PetriNet.Place("start")
    end = PetriNet.Place("end")
    transition = PetriNet.Transition("t_a", "A")

    net.places.update({start, end})
    net.transitions.add(transition)
    petri_utils.add_arc_from_to(start, transition, net)
    petri_utils.add_arc_from_to(transition, end, net)

    model = DiscoveredPetriNet("manual", {}, net, {}, {})
    return model, start, end, transition


class LocalReplayerTest(TestCase):
    def test_replay_consumes_input_and_produces_output_token(self):
        model, start, end, transition = _single_transition_model()
        participant = Participant("node-1")
        participant.places = [start, end]
        participant.transitions = [transition]
        participant.produce_token(start)

        LocalReplayer(model, participant).replay("A")

        self.assertFalse(participant.has_token(start))
        self.assertTrue(participant.has_token(end))
        self.assertEqual(
            participant.metrics,
            {"consumed_tokens": 1, "produced_tokens": 1},
        )

    def test_replay_counts_missing_token_and_continues(self):
        model, start, end, transition = _single_transition_model()
        participant = Participant("node-1")
        participant.places = [start, end]
        participant.transitions = [transition]

        LocalReplayer(model, participant).replay("A")

        self.assertEqual(participant.metrics["consumed_tokens"], 1)
        self.assertEqual(participant.metrics["produced_tokens"], 1)
        self.assertEqual(participant.metrics["missing_tokens"], 1)
        self.assertTrue(participant.has_token(end))

    def test_unknown_activity_is_counted_and_rejected(self):
        model, _, _, transition = _single_transition_model()
        participant = Participant("node-1")
        participant.transitions = [transition]

        with self.assertRaises(ValueError):
            LocalReplayer(model, participant).replay("missing")

        self.assertEqual(participant.metrics["unknown_activities"], 1)

    def test_replay_can_consume_remote_token_without_counting_missing_token(self):
        model, start, end, transition = _single_transition_model()
        owner = Participant("owner")
        owner.places = [start]
        owner.produce_token(start)
        participant = Participant("worker")
        participant.places = [end]
        participant.transitions = [transition]
        network = NetworkSimulator(
            {"owner": owner, "worker": participant},
            {"A": "worker"},
            {start: "owner", end: "worker"},
        )

        LocalReplayer(model, participant, network).replay("A")

        self.assertFalse(owner.has_token(start))
        self.assertTrue(participant.has_token(end))
        self.assertNotIn("missing_tokens", participant.metrics)
        self.assertEqual(participant.metrics["consumed_tokens"], 1)
        self.assertEqual(participant.metrics["produced_tokens"], 1)
        self.assertEqual(network.metrics()["remote_calls"], 1)
