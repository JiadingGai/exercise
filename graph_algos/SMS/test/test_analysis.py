from ..data import build_fig2
from ..analysis import *

class TestAnalysis:
    def test_asap(self):
      DFG, engineKind, hwConfig, Latency = build_fig2()
      ASAP = compute_asap(DFG, Latency, engineKind)
      ALAP = compute_alap(ASAP, DFG, Latency, engineKind)
      MOB = compute_mob(ALAP, ASAP)
      Depth = compute_depth(DFG, Latency, engineKind)
      Height = compute_height(DFG, Latency, engineKind)
      assert ASAP.__str__() == "{'n9': 0, 'n2': 0, 'n1': 0, 'n6': 2, 'n5': 2, 'n8': 4, 'n10': 6, 'n11': 8, 'n12': 10, 'n3': 2, 'n4': 4, 'n7': 6}"
