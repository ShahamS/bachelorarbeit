class Event:
    def __init__(self, case_id, activity, location, time):
        self.activity = activity
        self.location = location
        self.node_id = location
        self.time = time
        self.case_id = case_id

    def __str__(self):
        return f"[{self.case_id}, {self.activity}, {self.location}, {self.time}]"
