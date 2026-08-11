'''
Anhand ausgewähltem Process Discovery-Algorithmus und Parametern 
wird ein Petri-Netz erzeugt. Das Netz kann anschließend für 
Token-Based Replay genutzt werden.
'''
class DiscoveredPetriNet:
    def __init__(self, algorithm, parameters, net, initial_marking, 
                 final_marking,):
        self.algorithm = algorithm
        self.parameters = parameters
        self.net = net
        self.initial_marking = initial_marking # Startzustand des Petri-Netzes
        self.final_marking = final_marking # Endzustand des Petri-Netzes

# Zusätzliche Properties: places_count, transitions_count, arcs_count
# labeled_transitions_count, silent_transitions_count
# => werden für CSV gebraucht

    def places_count(self):
        return len(self.net.places)

    def transitions_count(self):
        return len(self.net.transitions)

    def arcs_count(self):
        return len(self.net.arcs)

    def labeled_transitions_count(self):
        return len([t for t in self.net.transitions if t.label is not None])

    def silent_transitions_count(self):
        return len([t for t in self.net.transitions if t.label is None])