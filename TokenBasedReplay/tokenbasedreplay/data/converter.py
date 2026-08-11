import pandas as pd
from pm4py.objects.log.obj import Event as Pm4PyEvent
from pm4py.objects.log.obj import EventLog as Pm4PyEventLog
from pm4py.objects.log.obj import Trace
from pm4py.util import constants

from .event import Event
from .event_log import EventLog


class Converter:
    #Übersetzt externe Event-Logs in die interne CheckMyFlow-Struktur
    def to_event_log(self, event_log, location_key=None, skip_events_without_location=True):
        '''Konvertiert einen PM4Py EventLog in einen lokalen `EventLog`.
        `location_key` bestimmt, welches Attribut eines PM4Py-Events als
        verteilter Knoten verwendet wird, z.B. `org:group`, `org:resource` oder
        `location`.
        '''

        converted_log = EventLog()
        for trace in event_log:
            case_id = self._case_id_from_trace(trace)
            for pm4py_event in trace:
                if skip_events_without_location and location_key and location_key not in pm4py_event:
                    continue
                converted_log.add_event(
                    self.to_event(pm4py_event, case_id, location_key=location_key)
                )
        return converted_log

    def to_event(self, pm4py_event, case_id, location_key=None):
        '''Konvertiert ein einzelnes PM4Py-Event.'''
        location = self._read_optional_value(pm4py_event, location_key, default="")
        return Event(
            case_id=case_id,
            activity=str(pm4py_event["concept:name"]),
            location=str(location),
            time=pm4py_event["time:timestamp"],
        )

    def from_event_log(self, event_log):
        '''
        Konvertiert den lokalen EventLog zurück in einen PM4Py EventLog
        '''

        pm4py_log = Pm4PyEventLog()
        for case_id, events in event_log.iter_traces():
            trace = Trace()
            trace.attributes[constants.CASE_CONCEPT_NAME] = case_id
            for event in events:
                trace.append(self.from_event(event))
            pm4py_log.append(trace)
        return pm4py_log

    def from_event(self, event):
        '''Konvertiert ein lokales Event in ein PM4Py-Event.'''

        pm4py_event = Pm4PyEvent()
        pm4py_event["concept:name"] = event.activity
        pm4py_event["time:timestamp"] = pd.Timestamp(event.time)
        pm4py_event["case:concept:name"] = event.case_id
        pm4py_event["location"] = event.location
        return pm4py_event

    def _case_id_from_trace(self, trace):
        '''
        Liest die Case-ID aus den Trace-Attributen
        '''

        if constants.CASE_CONCEPT_NAME in trace.attributes:
            return str(trace.attributes[constants.CASE_CONCEPT_NAME])
        return str(trace.attributes["concept:name"])

    def _read_optional_value(self, event, key, default):
        '''Liest ein optionales Event-Attribut mit Fallback.'''

        if key and key in event:
            return event[key]
        return default
