'''
Klasse zur Bereitstellung von Zugriffsmethoden auf ein entdecktes Petri-Netz
'''
class PetriNetAccess:
    def __init__(self, discovered_model):
        self.model = discovered_model
        self.net = discovered_model.net

    def places(self):
        return list(self.net.places)

    def transitions(self):
        return list(self.net.transitions)

    def input_places(self, transition):
        return [arc.source for arc in transition.in_arcs]

    def output_places(self, transition):
        return [arc.target for arc in transition.out_arcs]

    def transition_label(self, transition):
        return transition.label

    def is_silent(self, transition):
        return transition.label is None
