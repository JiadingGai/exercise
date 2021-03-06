#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List

#g = nx.read_adjlist('./sample.graph')
g = nx.DiGraph()

# Book example 22.4
#g.add_weighted_edges_from([("u", "v", 0.75), \
#                           ("v", "y", 0.75), \
#                           ("y", "x", 0.75), \
#                           ("u", "x", 0.75), \
#                           ("x", "v", 0.75), \
#                           ("w", "y", 0.75), \
#                           ("w", "z", 0.75), \
#                           ("z", "z", 0.75)  \
#                          ])

# Real diamond
#g.add_weighted_edges_from([("u", "x", 0.75), \
#                           ("u", "y", 0.75), \
#                           ("u", "z", 0.75), \
#                           ("x", "y", 0.75), \
#                           ("y", "w", 0.75), \
#                           ("z", "w", 0.75), \
#                          ])

# Book example 22.5(a)
#g.add_weighted_edges_from([("s", "z", 0.75), \
#                           ("s", "w", 0.75), \
#                           ("z", "y", 0.75), \
#                           ("z", "w", 0.75), \
#                           ("y", "x", 0.75), \
#                           ("x", "z", 0.75), \
#                           ("w", "x", 0.75), \
#                           ("t", "v", 0.75), \
#                           ("t", "u", 0.75), \
#                           ("v", "s", 0.75), \
#                           ("v", "w", 0.75), \
#                           ("u", "v", 0.75), \
#                           ("u", "t", 0.75), \
#                          ])

# Almost a tree to test using Kahn's algorithm for topo sorting.
g.add_weighted_edges_from([("A", "D", 0.75), \
                           ("D", "F", 0.75), \
                           ("B", "E", 0.75), \
                           ("E", "F", 0.75), \
                           ("E", "G", 0.75), \
                           ("C", "G", 0.75), \
                           ("F", "H", 0.75), \
                           ("G", "H", 0.75), \
                          ])

# Viz Method 1: use matplotlib
#plt.subplot(121)
#nx.draw(g, with_labels=True, font_weight='bold')
#plt.show()

# Begin DFS
nodes = g
color = {}
depth = {}
finish = {}
parent = {}
for node in nodes:
  print(node)
  color[node] = 'White'

time = 0

def DFS_visit(g, color, depth, finish, parent, node):
  global time
  color[node] = 'Gray'
  time = time + 1
  depth[node] = time
  for child in iter(g[node]):
      if color[child] == 'White':
          print("Visiting \n", child)
          parent[child] = node
          DFS_visit(g, color, depth, finish, parent, child)
  color[node] = 'Black'
  time = time + 1
  finish[node] = time

def DFS_visit_iter(g, color, depth, finish, parent, node):
    global time
    time = time + 1
    depth[node] = time
    color[node] = 'Gray'
    # Using list as a stack:
    # https://docs.python.org/2/tutorial/datastructures.html#using-lists-as-stacks
    stack = [(node, iter(list(g[node])))] # i.e., (v, adj(v))

    topo_order = []
    edge_type = {}
    while stack:
        cur_node, children = stack[-1]
        v = next(children, None)
        if v is None:
            time = time + 1
            color[cur_node] = 'Black'
            finish[cur_node] = time
            stack.pop()
            # Insert each vertex to the front of the topo_oder list as it
            # finishes.
            topo_order = [cur_node] + topo_order
            continue

        if (color[v] == 'White'):
            parent[v] = node
            time = time + 1
            depth[v] = time
            color[v] = 'Gray'
            stack.append((v, iter(list(g[v]))))
            edge_type[cur_node+"->"+v] = "tree edge"
        elif (color[v] == 'Gray'):
            # Detects a cycle because gray vertices are all ancestors of the
            # currently explored vertex (i.e., they're on the stack!!).
            print("WARNING: A cycle is in the graph! Topo order not defined!")
            edge_type[cur_node+"->"+v] = "back edge"
        elif (color[v] == 'Black'):
            edge_type[cur_node+"->"+v] = "forward or cross edge"

    return topo_order, edge_type

def topo_sorting_kahn(g):
    refCount = {n: g.in_degree(n) for n in g.nodes}
    print(refCount)
    nodeStack = list(reversed(sorted([n for n in g.nodes if g.out_degree(n) == 0])))
    TopoOrdering = []
    while len(nodeStack) > 0:
        n = nodeStack.pop()
        print("(INFO) pop ", n)
        if refCount[n] == 0:
            print("(INFO) ", n, " has zero ref count!")
            TopoOrdering.append(n)
            for succ in g.successors(n):
                refCount[succ] -= 1
                print("(INFO)     decrement successor: ", succ, ", it has ", refCount[succ])
            print("(INFO) topo ordering so far = ", TopoOrdering)
        else:
            nodeStack.append(n)
            predNodes = list(reversed(sorted([pred for pred in g.predecessors(n) if pred not in TopoOrdering])))
            nodeStack += predNodes
            print("(INFO) add ", n, " and predecessors: ", predNodes, " back to nodeStack = ", nodeStack)
    # Check if all the nodes in g is in the topo ordering.
    assert len(TopoOrdering) == len(g.nodes)
    # No duplicated nodes in the topo ordering
    assert len(set(TopoOrdering)) == len(TopoOrdering)

    return TopoOrdering


print(topo_sorting_kahn(g))

for node in nodes:
    if color[node] == 'White':
        print("Visiting \n - top level", node)
        #DFS_visit(g, color, depth, finish, parent, node)
        topo_order, edge_type = DFS_visit_iter(g, color, depth, finish, parent, node)
        print('Topological order starting at [', node, "] is: \n    ", topo_order)
        print('Edge types: ', edge_type)
print("Depth: ", depth)
print("Finish: ", finish)
print("Parent: ", parent)

print("\n[Golden Reference]:")
print("""list(nx.dfs_edges(g, source="u"))""")
print(list(nx.dfs_edges(g, source="u")))

print("""list(nx.dfs_tree(g, source="u"))""")
print(list(nx.dfs_tree(g, source="u")))

print("""list(nx.dfs_labeled_edges(g, source="u"))""")
print(list(nx.dfs_labeled_edges(g, source="u")))
# Viz Method 2: use graphviz
#h = nx.nx_agraph.from_agraph(g)
#nx.write_dot(h, './graph.dot')
