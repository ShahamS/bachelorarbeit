import random
from pathlib import Path

import pm4py

from .converter import Converter
from .event_log import EventLog


class EventLogSplitter:
    '''
    Lädt optional ein XES-Log und splittet es auf Trace-Ebene.
    Nah an distributed_alignemnts; Unterschied ist, dass
    die Split-Rate aus dem Parameter genutzt wird.
    '''

    def __init__(self, file_path=None, event_log=None, training_split=0.8, location_key="org:group", random_seed=1):
        self.converter = Converter()
        self.training_split = training_split
        self.location_key = location_key
        self.random_seed = random_seed
        self.log = event_log if event_log is not None else self._read_xes_log(file_path, location_key)
        self.case_ids = self._get_case_ids()
        self.split_index = self._calculate_split_index()

    def _read_xes_log(self, file_path, location_key):
        pm4py_log = pm4py.read_xes(str(Path(file_path)), return_legacy_log_object=True)
        return self.converter.to_event_log(pm4py_log, location_key=location_key)

    def _get_case_ids(self):
        return self.log.case_ids()

    def _calculate_split_index(self):
        random.Random(self.random_seed).shuffle(self.case_ids)
        return int(len(self.case_ids) * self.training_split)

    def split(self):
        return self.get_training_data(), self.get_test_data()

    def get_training_data(self, number_of_records=None):
        '''Gibt Trainingsdaten zurück.
        Ohne `number_of_records` wird die konfigurierte Split-Grenze verwendet.
        Mit `number_of_records` kann man für Experimente gezielt weniger Traces
        aus dem Trainingsbereich nehmen.
        '''

        end = self.split_index if number_of_records is None else number_of_records
        return self.log.filter_case_ids(self.case_ids[:end])

    def get_test_data(self, record_index=None):
        '''
        Gibt Testdaten zurück.
        Ohne `record_index` werden alle Test-Traces geliefert. Mit Index wird
        genau ein Test-Trace ausgewählt, ähnlich wie im Vorbildcode.
        '''

        test_case_ids = self.case_ids[self.split_index :]
        if record_index is None:
            return self.log.filter_case_ids(test_case_ids)
        return self.log.filter_case_ids([test_case_ids[record_index]])

    def get_case(self, case_id):
        '''
        Gibt einen einzelnen Case als EventLog zurück.
        '''

        return self.log.filter_case_ids([case_id])
