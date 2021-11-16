#!/usr/env/python3
# Implementation of Swing Modulo Scheduling based on
# "Lifetime-sensitive Modulo Scheduling in a Production Environment", Josep Llosa et al.
import networkx as nx
from analysis import *
from data import build_fig2

DFG, engineKind, hwConfig, Latency = build_fig2()

ASAP = compute_asap(DFG, Latency, engineKind)
ALAP = compute_alap(ASAP, DFG, Latency, engineKind)
MOB = compute_mob(ALAP, ASAP)
Depth = compute_depth(DFG, Latency, engineKind)
Height = compute_height(DFG, Latency, engineKind)
print(Height)

