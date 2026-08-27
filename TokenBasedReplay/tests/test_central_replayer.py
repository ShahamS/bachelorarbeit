from unittest import TestCase

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from tokenbasedreplay.data.event import Event
from tokenbasedreplay.data.event_log import EventLog
from tokenbasedreplay.model.central.central_replayer import CentralTokenReplayer
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

    initial_marking = Marking()
    initial_marking[start] = 1
    final_marking = Marking()
    final_marking[end] = 1

    return DiscoveredPetriNet("manual", {}, net, initial_marking, final_marking)


class CentralTokenReplayerTest(TestCase):
    def test_matching_trace_has_perfect_fitness(self):
        model = _single_transition_model()
        log = EventLog()
        log.add_event(Event("case-1", "A", "central", 1))

        result = CentralTokenReplayer(model).replay_log(log)

        self.assertEqual(result.total_events, 1)
        self.assertEqual(result.total_consumed_tokens, 1)
        self.assertEqual(result.total_produced_tokens, 1)
        self.assertEqual(result.total_missing_tokens, 0)
        self.assertEqual(result.total_remaining_tokens, 0)
        self.assertEqual(result.fitness, 1.0)

    def test_each_trace_starts_with_fresh_initial_marking(self):
        model = _single_transition_model()
        log = EventLog()
        log.add_event(Event("case-1", "A", "central", 1))
        log.add_event(Event("case-2", "A", "central", 2))

        result = CentralTokenReplayer(model).replay_log(log)

        self.assertEqual(result.total_events, 2)
        self.assertEqual(result.total_missing_tokens, 0)
        self.assertEqual(result.fitness, 1.0)

    def test_missing_token_is_counted_but_replay_continues(self):
        model = _single_transition_model()
        model.initial_marking.clear()
        log = EventLog()
        log.add_event(Event("case-1", "A", "central", 1))

        result = CentralTokenReplayer(model).replay_log(log)

        self.assertEqual(result.total_missing_tokens, 1)
        self.assertEqual(result.total_produced_tokens, 1)
        self.assertEqual(result.total_remaining_tokens, 0)
        self.assertLess(result.fitness, 1.0)

    def test_unknown_activity_is_recorded(self):
        model = _single_transition_model()
        log = EventLog()
        log.add_event(Event("case-1", "unknown", "central", 1))

        result = CentralTokenReplayer(model).replay_log(log)

        self.assertEqual(result.unknown_activities, 1)
        self.assertEqual(result.matched_events, 0)
