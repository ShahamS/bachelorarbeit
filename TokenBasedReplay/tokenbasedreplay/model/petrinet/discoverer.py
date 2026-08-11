from pm4py.discovery import discover_petri_net_alpha, discover_petri_net_heuristics, discover_petri_net_inductive

from .discovered_model import DiscoveredPetriNet
'''
Hier werden die Petri-Netze anhand der Process-Discovery-Algorithmen aus
PM4Py erzeugt. Anschließend werden die Netze in das lokale 
DiscoveredPetriNet-Format konvertiert.
'''
class DiscoveryRunner:
    def discover_alpha(self, pm4py_log, activity_key: str = 'concept:name',
                       case_id_key: str = 'case:concept:name',
                       timestamp_key: str = 'time:timestamp') -> DiscoveredPetriNet:

        net, im, fm = discover_petri_net_alpha(
            pm4py_log, activity_key=activity_key,
            case_id_key=case_id_key,
            timestamp_key=timestamp_key)

        return DiscoveredPetriNet(
            algorithm='alpha',
            parameters={},
            net=net,
            initial_marking=im,
            final_marking=fm
        )

    # Heuristics-Algorithmus muss Dependency-Threshold als Parameter bekommen
    def discover_heuristics(self, pm4py_log, dependency_threshold: float, activity_key: str = 'concept:name',
                            case_id_key: str = 'case:concept:name',
                            timestamp_key: str = 'time:timestamp') -> DiscoveredPetriNet:

        net, im, fm = discover_petri_net_heuristics(
            pm4py_log, activity_key=activity_key,
            case_id_key=case_id_key,
            timestamp_key=timestamp_key,
            dependency_threshold=dependency_threshold
        )

        return DiscoveredPetriNet(
            algorithm='heuristics',
            parameters={'dependency_threshold': dependency_threshold},
            net=net,
            initial_marking=im,
            final_marking=fm
        )

    def discover_inductive(self, pm4py_log, activity_key: str = 'concept:name',
                           case_id_key: str = 'case:concept:name',
                           timestamp_key: str = 'time:timestamp') -> DiscoveredPetriNet:

        net, im, fm = discover_petri_net_inductive(
            pm4py_log, activity_key=activity_key,
            case_id_key=case_id_key,
            timestamp_key=timestamp_key
        )

        return DiscoveredPetriNet(
            algorithm='inductive',
            parameters={},
            net=net,
            initial_marking=im,
            final_marking=fm
        )

    # Man kann mehrere Threshold für den Heuristics-Algorithmus angeben, um mehrere Netze zu erzeugen
    def discover_all(self, pm4py_log, heuristic_thresholds=None, activity_key: str = 'concept:name',
                     case_id_key: str = 'case:concept:name',
                     timestamp_key: str = 'time:timestamp') -> list[DiscoveredPetriNet]:

        discovered_nets = []
        discovered_nets.append(self.discover_alpha(pm4py_log, activity_key=activity_key, case_id_key=case_id_key, timestamp_key=timestamp_key))
        discovered_nets.append(self.discover_inductive(pm4py_log, activity_key=activity_key, case_id_key=case_id_key, timestamp_key=timestamp_key))
        if heuristic_thresholds is not None:
            for threshold in heuristic_thresholds:
                discovered_nets.append(self.discover_heuristics(pm4py_log, threshold, activity_key=activity_key, case_id_key=case_id_key, timestamp_key=timestamp_key))
        return discovered_nets
