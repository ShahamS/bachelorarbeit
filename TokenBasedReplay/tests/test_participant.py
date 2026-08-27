from unittest import TestCase

from tokenbasedreplay.model.participant.participant import Participant


class ParticipantTest(TestCase):
    def test_has_token_checks_requested_place(self):
        participant = Participant("node-1")

        participant.produce_token("p1")

        self.assertTrue(participant.has_token("p1"))
        self.assertFalse(participant.has_token("p2"))

    def test_consume_token_and_finalize_remaining_update_metrics(self):
        participant = Participant("node-1")

        participant.produce_token("p1")
        participant.produce_token("p1")
        participant.consume_token("p1")
        participant.finalize_remaining()

        self.assertTrue(participant.has_token("p1"))
        self.assertEqual(participant.get_metrics(), {"remaining_tokens": 1})
