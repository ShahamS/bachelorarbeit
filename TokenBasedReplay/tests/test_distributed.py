from unittest import TestCase

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from tokenbasedreplay.data.event import Event
from tokenbasedreplay.data.event_log import EventLog
from tokenbasedreplay.model.central.central_replayer import CentralTokenReplayer
from tokenbasedreplay.model.distributed.assignment import ParticipantAssignment
from tokenbasedreplay.model.distributed.distributed_replayer import DistributedTokenReplayer
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

    initial_marking = Marking()
    initial_marking[start] = 1
    final_marking = Marking()
    final_marking[end] = 1

    model = DiscoveredPetriNet("manual", {}, net, initial_marking, final_marking)
    return model, start, end, transition


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

    model = DiscoveredPetriNet("manual", {}, net, initial_marking, final_marking)
    return model, start, middle, end, transition_a, transition_b


class ParticipantAssignmentTest(TestCase):
    def test_build_reuses_participants_and_returns_ids_for_activity_mapping(self):
        model, start, end, transition = _single_transition_model()
        training_log = EventLog()
        training_log.add_event(Event("case-1", "A", "node-1", 1))
        training_log.add_event(Event("case-2", "A", "node-1", 2))

        participants, activity_mapping, place_mapping = ParticipantAssignment(
            model,
            training_log,
        ).build()

        self.assertEqual(list(participants.keys()), ["node-1"])
        self.assertEqual(activity_mapping, {"A": "node-1"})
        self.assertEqual(participants["node-1"].transitions, [transition])
        self.assertEqual(participants["node-1"].places, [start, end])
        self.assertTrue(participants["node-1"].has_token(start))
        self.assertEqual(place_mapping[start], "node-1")


class DistributedTokenReplayerTest(TestCase):
    def test_replay_log_returns_result_and_counts_route_calls(self):
        model, _, _, _ = _single_transition_model()
        training_log = EventLog()
        training_log.add_event(Event("case-1", "A", "node-1", 1))
        participants, activity_mapping, place_mapping = ParticipantAssignment(
            model,
            training_log,
        ).build()
        network = NetworkSimulator(participants, activity_mapping, place_mapping)
        test_log = EventLog()
        test_log.add_event(Event("case-2", "A", "node-1", 2))

        result = DistributedTokenReplayer(model, participants, network).replay_log(test_log)

        self.assertEqual(result.total_events, 1)
        self.assertEqual(result.matched_events, 1)
        self.assertEqual(result.missing_tokens, 0)
        self.assertEqual(result.total_consumed_tokens, 1)
        self.assertEqual(result.total_produced_tokens, 1)
        self.assertEqual(result.total_missing_tokens, 0)
        self.assertEqual(result.total_remaining_tokens, 0)
        self.assertEqual(result.fitness, 1.0)
        self.assertEqual(result.network_metrics["route_calls"], 1)
        self.assertEqual(result.network_metrics["local_calls"], 1)
        self.assertEqual(result.local_calls, 1)
        self.assertEqual(result.remote_calls, 0)
        self.assertEqual(result.route_calls, 1)

    def test_replay_log_aggregates_missing_remaining_and_fitness(self):
        model, _, _, _ = _single_transition_model()
        model.initial_marking.clear()
        training_log = EventLog()
        training_log.add_event(Event("case-1", "A", "node-1", 1))
        participants, activity_mapping, place_mapping = ParticipantAssignment(
            model,
            training_log,
        ).build()
        network = NetworkSimulator(participants, activity_mapping, place_mapping)
        test_log = EventLog()
        test_log.add_event(Event("case-2", "A", "node-1", 2))

        result = DistributedTokenReplayer(model, participants, network).replay_log(test_log)

        self.assertEqual(result.total_consumed_tokens, 1)
        self.assertEqual(result.total_produced_tokens, 1)
        self.assertEqual(result.total_missing_tokens, 1)
        self.assertEqual(result.total_remaining_tokens, 0)
        self.assertEqual(result.fitness, 0.5)

    def test_each_case_starts_with_fresh_distributed_initial_marking(self):
        model, _, _, _ = _single_transition_model()
        training_log = EventLog()
        training_log.add_event(Event("training", "A", "node-1", 1))
        participants, activity_mapping, place_mapping = ParticipantAssignment(
            model,
            training_log,
        ).build()
        network = NetworkSimulator(participants, activity_mapping, place_mapping)
        test_log = EventLog()
        test_log.add_event(Event("case-1", "A", "node-1", 2))
        test_log.add_event(Event("case-2", "A", "node-1", 3))

        result = DistributedTokenReplayer(model, participants, network).replay_log(test_log)

        self.assertEqual(result.total_events, 2)
        self.assertEqual(result.total_consumed_tokens, 2)
        self.assertEqual(result.total_produced_tokens, 2)
        self.assertEqual(result.total_missing_tokens, 0)
        self.assertEqual(result.total_remaining_tokens, 0)
        self.assertEqual(result.fitness, 1.0)

    def test_cross_participant_trace_consumes_remote_case_token(self):
        model, _, _, _, _, _ = _two_transition_model()
        training_log = EventLog()
        training_log.add_event(Event("training", "A", "node-1", 1))
        training_log.add_event(Event("training", "B", "node-2", 2))
        participants, activity_mapping, place_mapping = ParticipantAssignment(
            model,
            training_log,
        ).build()
        network = NetworkSimulator(participants, activity_mapping, place_mapping)
        test_log = EventLog()
        test_log.add_event(Event("case-1", "A", "node-1", 3))
        test_log.add_event(Event("case-1", "B", "node-2", 4))

        result = DistributedTokenReplayer(model, participants, network).replay_log(test_log)

        self.assertEqual(result.total_missing_tokens, 0)
        self.assertEqual(result.total_remaining_tokens, 0)
        self.assertEqual(result.fitness, 1.0)
        self.assertEqual(result.remote_calls, 1)
        self.assertEqual(result.network_metrics["remote_token_checks"], 1)
        self.assertEqual(result.network_metrics["local_token_checks"], 1)

    def test_failed_remote_token_request_counts_remote_check_and_missing_token(self):
        model, _, _, _, _, _ = _two_transition_model()
        training_log = EventLog()
        training_log.add_event(Event("training", "A", "node-1", 1))
        training_log.add_event(Event("training", "B", "node-2", 2))
        participants, activity_mapping, place_mapping = ParticipantAssignment(
            model,
            training_log,
        ).build()
        network = NetworkSimulator(participants, activity_mapping, place_mapping)
        test_log = EventLog()
        test_log.add_event(Event("case-1", "B", "node-2", 3))

        result = DistributedTokenReplayer(model, participants, network).replay_log(test_log)

        self.assertEqual(result.total_consumed_tokens, 1)
        self.assertEqual(result.total_missing_tokens, 1)
        self.assertEqual(result.remote_calls, 1)
        self.assertEqual(result.network_metrics["remote_token_checks"], 1)

    def test_unknown_event_without_routing_target_is_recorded_without_route_call(self):
        model, _, _, _ = _single_transition_model()
        training_log = EventLog()
        training_log.add_event(Event("training", "A", "node-1", 1))
        participants, activity_mapping, place_mapping = ParticipantAssignment(
            model,
            training_log,
        ).build()
        network = NetworkSimulator(participants, activity_mapping, place_mapping)
        test_log = EventLog()
        test_log.add_event(Event("case-1", "unknown", None, 2))

        result = DistributedTokenReplayer(model, participants, network).replay_log(test_log)

        self.assertEqual(result.total_events, 1)
        self.assertEqual(result.unknown_activities, 1)
        self.assertEqual(result.route_calls, 0)
        self.assertEqual(result.remote_calls, 0)
        self.assertEqual(result.local_calls, 0)

    def test_single_participant_result_matches_central_replay(self):
        model, _, _, _, _, _ = _two_transition_model()
        training_log = EventLog()
        training_log.add_event(Event("training", "A", "node-1", 1))
        training_log.add_event(Event("training", "B", "node-1", 2))
        participants, activity_mapping, place_mapping = ParticipantAssignment(
            model,
            training_log,
        ).build()
        network = NetworkSimulator(participants, activity_mapping, place_mapping)
        test_log = EventLog()
        test_log.add_event(Event("case-1", "A", "node-1", 3))
        test_log.add_event(Event("case-1", "B", "node-1", 4))

        distributed_result = DistributedTokenReplayer(
            model,
            participants,
            network,
        ).replay_log(test_log)
        central_result = CentralTokenReplayer(model).replay_log(test_log)

        self.assertEqual(distributed_result.total_consumed_tokens, central_result.total_consumed_tokens)
        self.assertEqual(distributed_result.total_produced_tokens, central_result.total_produced_tokens)
        self.assertEqual(distributed_result.total_missing_tokens, central_result.total_missing_tokens)
        self.assertEqual(distributed_result.total_remaining_tokens, central_result.total_remaining_tokens)
        self.assertEqual(distributed_result.fitness, central_result.fitness)
