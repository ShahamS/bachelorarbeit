from unittest import TestCase

from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from tokenbasedreplay.model.petrinet.discovered_model import DiscoveredPetriNet
from tokenbasedreplay.model.petrinet.marking import LocalMarking
from tokenbasedreplay.model.petrinet.net_access import PetriNetAccess
from tokenbasedreplay.model.petrinet.transition_lookup import TransitionLookup

'''
Testet die Hilfsklassen für den Zugriff auf Petri-Netze, wie PetriNetAccess und 
TransitionLookup. Die Tests prüfen, ob die Methoden korrekt die Plätze, Transitionen, 
Kanten und Labels des Petri-Netzes zurückgeben und ob die Markierung korrekt verwaltet 
wird.
'''

def _mini_net():
    net = PetriNet("mini")
    start = PetriNet.Place("start")
    middle = PetriNet.Place("middle")
    end = PetriNet.Place("end")
    transition_a = PetriNet.Transition("t_a", "A")
    transition_b = PetriNet.Transition("t_b", "B")
    silent_transition = PetriNet.Transition("t_silent", None)

    net.places.update({start, middle, end})
    net.transitions.update({transition_a, transition_b, silent_transition})
    petri_utils.add_arc_from_to(start, transition_a, net)
    petri_utils.add_arc_from_to(transition_a, middle, net)
    petri_utils.add_arc_from_to(middle, transition_b, net)
    petri_utils.add_arc_from_to(transition_b, end, net)

    return net, start, middle, end, transition_a, transition_b, silent_transition


class LocalMarkingTest(TestCase):
    def test_add_get_remove_and_remaining_tokens(self):
        marking = LocalMarking()

        marking.add("p1")
        marking.add("p1", 2)
        marking.add("p2")
        marking.remove("p1")

        self.assertEqual(marking.get("p1"), 2)
        self.assertEqual(marking.get("p2"), 1)
        self.assertEqual(marking.get("unknown"), 0)
        self.assertEqual(marking.remaining_tokens(), 3)

    def test_remove_deletes_place_when_amount_reaches_zero(self):
        marking = LocalMarking()

        marking.add("p1", 2)
        marking.remove("p1", 2)
        marking.remove("unknown")

        self.assertEqual(marking.get("p1"), 0)
        self.assertEqual(marking.remaining_tokens(), 0)


class PetriNetAccessTest(TestCase):
    def test_access_returns_places_transitions_labels_and_silent_flag(self):
        net, start, middle, end, transition_a, transition_b, silent_transition = _mini_net()
        model = DiscoveredPetriNet("manual", {}, net, {}, {})
        access = PetriNetAccess(model)

        self.assertEqual(set(access.places()), {start, middle, end})
        self.assertEqual(set(access.transitions()), {transition_a, transition_b, silent_transition})
        self.assertEqual(access.input_places(transition_a), [start])
        self.assertEqual(access.output_places(transition_a), [middle])
        self.assertEqual(access.transition_label(transition_a), "A")
        self.assertFalse(access.is_silent(transition_a))
        self.assertTrue(access.is_silent(silent_transition))


class TransitionLookupTest(TestCase):
    def test_find_transition_uses_activity_label_and_ignores_silent_transitions(self):
        net, _, _, _, transition_a, _, _ = _mini_net()
        lookup = TransitionLookup(net)

        self.assertIs(lookup.find_transition("A"), transition_a)
        self.assertIsNone(lookup.find_transition("missing"))
        self.assertEqual(lookup.find_transitions("missing"), [])

    def test_ambiguous_labels_are_reported(self):
        net, _, _, _, transition_a, _, _ = _mini_net()
        duplicate_a = PetriNet.Transition("t_a_duplicate", "A")
        net.transitions.add(duplicate_a)

        lookup = TransitionLookup(net)

        self.assertEqual(lookup.find_transitions("A"), [transition_a, duplicate_a])
        self.assertEqual(lookup.ambiguous_labels(), {"A": [transition_a, duplicate_a]})
