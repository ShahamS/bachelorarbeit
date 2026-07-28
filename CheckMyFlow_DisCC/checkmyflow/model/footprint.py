from collections import defaultdict


START_ACTIVITY = "__START__"


class FootprintMatrix:
    # Speichert erlaubte direkte Vorgänger-Nachfolger-Beziehungen

    def __init__(self):
        self.allowed_predecessors = defaultdict(set)

    def add_direct_succession(self, previous_activity, current_activity):
        # Lernt, dass `current_activity` nach `previous_activity` auftreten darf

        self.allowed_predecessors[current_activity].add(previous_activity)

    def add_start_activity(self, activity):
        # Lernt, dass eine Aktivität am Trace-Anfang auftreten darf

        self.add_direct_succession(START_ACTIVITY, activity)

    def allows(self, previous_activity, current_activity):
        # Prüft, ob die direkte Nachfolge im Modell bekannt ist

        return previous_activity in self.allowed_predecessors.get(current_activity, set())

    def allowed_for(self, current_activity):
        # Gibt alle erlaubten Vorgänger für eine Aktivität zurück

        return set(self.allowed_predecessors.get(current_activity, set()))

    def activities(self):
        # Gibt alle Aktivitäten zurück, für die Vorgänger gelernt wurden

        return set(self.allowed_predecessors.keys())

    def merge(self, other):
        # Übernimmt alle Relationen aus einer anderen FootprintMatrix

        for current_activity, predecessors in other.allowed_predecessors.items():
            self.allowed_predecessors[current_activity].update(predecessors)

    def to_dict(self):
        # Gibt eine normale Dict-Kopie für Debugging oder Tests zurück

        return {
            current_activity: set(predecessors)
            for current_activity, predecessors in self.allowed_predecessors.items()
        }

    @classmethod
    def from_pairs(cls, pairs):
        # Erzeugt eine FootprintMatrix aus `(previous, current)`-Paaren

        matrix = cls()
        for previous_activity, current_activity in pairs:
            matrix.add_direct_succession(previous_activity, current_activity)
        return matrix
