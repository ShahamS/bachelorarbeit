from unittest import TestCase

from tokenbasedreplay.data.event import Event
from tokenbasedreplay.data.event_log import EventLog


class EventLogTest(TestCase):
    def test_filter_case_ids_accepts_generators(self):
        log = EventLog()
        log.add_event(Event("case-1", "A", "node-1", 1))
        log.add_event(Event("case-2", "B", "node-2", 2))
        log.add_event(Event("case-3", "C", "node-3", 3))

        filtered = log.filter_case_ids(case_id for case_id in ["case-1", "case-3"])

        self.assertEqual(filtered.case_ids(), ["case-1", "case-3"])
