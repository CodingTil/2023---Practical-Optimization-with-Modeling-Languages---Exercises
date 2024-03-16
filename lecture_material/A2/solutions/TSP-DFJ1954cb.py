# -*- coding: utf-8 -*-
"""
solve a TSPLIB input file via Dantzig, Fulkerson, Johnson (1954)

uses so-called "callbacks" to "lazily" add SECs in every node of the B&C tree

@author: luebbecke

version: 2020pom
"""


import sys
import re
import math
from gurobipy import *
import networkx as nx  # a library with network algorithms
import matplotlib

matplotlib.use("TkAgg")  # opens a windows for plot()
import matplotlib.pyplot as plt


###############################################################################
def visu_LP(cities, x):
    plt.plot(cities[0], cities[1], "o")
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            if x[i, j].x > 0.0001:
                plt.plot(
                    [cities[0][i], cities[0][j]],
                    [cities[1][i], cities[1][j]],
                    color="black",
                    alpha=x[i, j].x,
                    linewidth=x[i, j].x,
                )
    # plt.savefig("st70cb.png", dpi=300)
    plt.show()


###############################################################################

# a little help message if called without argument
if len(sys.argv) != 2:
    print(
        "usage:\n%s <TSPLIB input file in EDGE_WEIGHT_TYPE: EUC_2D format>"
        % (sys.argv[0].split("/")[-1])
    )
    sys.exit(1)

# an elementary example for try/except exception handling
try:
    infile = open(str(sys.argv[1]), "r")
except:
    print("File " + str(sys.argv[1]) + " not found.")
    sys.exit(1)

# the TSPLIB file format is described in this document:
# https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf


cities = [[], []]
for line in infile:
    if line.find("EDGE_WEIGHT_TYPE:") > -1 and line.find("EUC_2D") == -1:
        print("Unsupported EDGE_WEIGHT_TYPE\n")
        sys.exit(1)
    if line.find("DIMENSION") > -1:
        n = int(line.split(":")[1])
        continue
    if re.match("[a-zA-Z]", line) != None or len(line) < 5:
        continue
    # this must be a coordinate of a city
    c = int(line.split()[0])
    xcoord = float(line.split()[1])
    ycoord = float(line.split()[2])
    cities[0].append(xcoord)
    cities[1].append(ycoord)


# print (cities)
nodes = range(n)


# the main model ##############################################################
model = Model("TSP-DFJcb")

# undirected graph; edge variables: edge is in tour?
x = {}
for i in nodes:
    for j in nodes:
        if i >= j:
            continue
        x[i, j] = model.addVar(
            name="x_" + str(i) + "_" + str(j),
            vtype="b",
            # the format requires rounding to nearest integer
            obj=round(
                math.sqrt(
                    (cities[0][i] - cities[0][j]) ** 2
                    + (cities[1][i] - cities[1][j]) ** 2
                )
            ),
        )

# the objective is to minimize the tour length
model.modelSense = GRB.MINIMIZE

# degree 2 equalities
for i in nodes:
    model.addConstr(
        quicksum(x[i, j] for j in nodes if i < j)
        + quicksum(x[j, i] for j in nodes if j < i)
        == 2
    )

model.write("TSPDFJcb.lp")
SECs_added = 0


###############################################################################


# define a so-called "callback" which in each node of the B&C tree (not only
# at the root node) adds violated subtour elimination constraints


def separateSEC(model, where):

    global SECs_added

    # from the gurobi documentation:
    # "MIPSOL" is called after a new incumbent is found
    # "MIPNODE" is called when a node was solved, also once for each cut pass during the root node solve
    if where == GRB.Callback.MIPNODE or where == GRB.Callback.MIPSOL:

        if where == GRB.Callback.MIPSOL:
            rel = model.cbGetSolution(x)
            # print("checking an integer solution")

        else:  # i.e., where == MIPNODE
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status != GRB.OPTIMAL:
                return

            rel = model.cbGetNodeRel(x)
            # print("checking a fractional solution")

        # check for violated SEC (min cut algorithm)
        G = nx.Graph()
        for i in nodes:
            for j in nodes:
                if i >= j:
                    continue
                G.add_edge(
                    i, j, weight=max(0, rel[i, j])
                )  # max to avoid (small) negative values
        cut_val, partition = nx.stoer_wagner(G)

        # if found violated SEC, add it to the model
        if cut_val < 1.99999:  # for numerical reasons, "sufficiently" smaller than 2.0
            violated_cut = []
            for i in partition[0]:
                for j in partition[1]:
                    if i < j:
                        violated_cut.append((i, j))
                    else:
                        violated_cut.append((j, i))
            model.cbLazy(quicksum(x[i, j] for (i, j) in violated_cut) >= 2)
            SECs_added += 1


# indicate that some constraints are "lazily" added to the model
model.params.LazyConstraints = 1

# solve (this starts the B&C algorithm, not only the LP relaxation)
model.optimize(separateSEC)

# visualization of solution
print("Added", SECs_added, "SECs.")
visu_LP(cities, x)
