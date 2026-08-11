class EventLog:
    def __init__(self, traces=None):
        if traces:
            self.traces = traces
        else:
            self.traces = {}

    def add_event(self, event):
        if event.case_id not in self.traces:
            self.traces[event.case_id] = []
        self.traces[event.case_id].append(event)

    def add_events(self, events):
        # Fügt mehrere Events ein und nutzt dabei dieselbe Trace-Gruppierung
        for event in events:
            self.add_event(event)

    def has_event_in_case(self, case_id, activity):
        if case_id not in self.traces:
            return False
        return True in [event.activity == activity for event in self.traces[case_id]]

    def filter_case_ids(self, case_ids):
            # Erzeuge ein neues Eventlog mit ausgewählten Case IDs
            new_event_log = EventLog()
            for case_id in self.traces:
                if case_id in case_ids:
                    for event in self.traces[case_id]:
                        new_event_log.add_event(event)
            return new_event_log

    def case_ids(self):
        # Gibt alle Case-IDs in Einfüge-Reihenfolge zurück
        return list(self.traces.keys())

    def iter_traces(self):
        # Iteriert über `(case_id, events)`-Paare
        return iter(self.traces.items())

    def __len__(self):
        # Anzahl der Traces, nicht Anzahl der Events
        return len(self.traces)
