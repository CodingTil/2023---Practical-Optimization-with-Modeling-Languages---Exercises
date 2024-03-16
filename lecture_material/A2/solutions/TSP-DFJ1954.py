# -*- coding: utf-8 -*-
"""
solve a TSPLIB input file via Dantzig, Fulkerson, Johnson (1954)

@author: luebbecke
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

# here is a different visualization using networkx' plot functionalities


def nx_visu_LP(cities, draw_x, x, partition, it):
    """

    Parameters
    ----------
    cities : two lists of doubles, contains the x- and y-coordinates of cities
    draw_x : Boolean to indicate whether edge labels (weights) should taken from support graph (False) or from the LP solution (True)
    x : gurobi variables, current solution
    partition : two node sets, as returned by Stoer/Wagner
    it : an integer counter, mainly for producing the correct file name

    Returns
    -------
    None.

    """

    plt.figure(figsize=(16, 9))  # size of image

    # data needed by the graph plotting methods
    coord = list(zip(cities[0], cities[1]))
    pos = dict(zip(nodes, coord))  # node coordinates
    edgelist = []  # sublist of edges with positive weight
    edgelabels = {}  # weights of the edges (= corresponding x variable values)
    linewidths = []  # draw thicker when variable value larger
    colors = []  # allows us to draw fractional edges in a special color
    for e in G.edges:
        if draw_x == False:
            w = G.get_edge_data(e[0], e[1])["weight"]
        else:
            w = x[e[0], e[1]].x
        if w > 0.001:
            edgelist.append(e)
            if w > 0.99:
                colors.append("black")
            else:
                colors.append("orange")
            edgelabels[e] = w
            linewidths = 2 * w

    # draw one side of the cut
    nodecolors = ["blue" for i in nodes]
    for i in partition:
        nodecolors[i] = "yellowgreen"

    # yes, believe me, I also checked the documentation to get this right
    nx.draw(
        G,
        pos,
        edgelist=edgelist,
        width=linewidths,
        edge_color=colors,
        node_color=nodecolors,
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edgelabels, label_pos=0.5, font_size=14
    )

    # we can save the images in any format, pdf was convenient to produce the slides
    plt.savefig(
        "tsp-%s-%04d.pdf" % (sys.argv[1].split(".")[0], it), dpi=300, transparent=True
    )
    # plt.show()


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
model = Model("TSP-DFJ")

# undirected graph; edge variables: edge is in tour?
x = {}
for i in nodes:
    for j in nodes:
        if i >= j:
            continue
        x[i, j] = model.addVar(
            name="x_" + str(i) + "_" + str(j),
            vtype="c",
            ub=1.0,  # you may change the vtype to GRB.BINARY here!
            # the format requires rounding to nearest integer!!
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

model.write("TSPDFJ.lp")

###############################################################################


# iteratively add violated subtour elimination constraints
SEC_violated = True
SECs_added = 0

while SEC_violated == True:
    # solve LP relaxation

    model.optimize()

    # check for violated SEC (min cut algorithm)
    G = nx.Graph()
    for i in nodes:
        for j in nodes:
            if i >= j:
                continue
            G.add_edge(i, j, weight=max(0, x[i, j].x))

    cut_val, partition = nx.stoer_wagner(G)

    # if found violated  SEC, add it to the model, and iterate
    if cut_val < 1.99999:  # for numerical reasons, "sufficiently" smaller than 2.0
        violated_cut = []
        for i in partition[0]:
            for j in partition[1]:
                if i < j:
                    violated_cut.append((i, j))
                else:
                    violated_cut.append((j, i))
        model.addConstr(quicksum(x[i, j] for (i, j) in violated_cut) >= 2)
        SEC_violated = True
        SECs_added += 1

        nx_visu_LP(cities, False, x, partition[0], SECs_added)

    # otherwise: LP relaxation is solved to optimality
    else:
        SEC_violated = False
        nx_visu_LP(
            cities, False, x, partition[0], SECs_added + 1
        )  # one plot of the final LP, needs x because we did not construct a support graph


# revert to binary variables
for v in model.getVars():
    v.vType = GRB.BINARY

# final run to get an integral solution
model.optimize()


# visualization of final solution

print("Added", SECs_added, "SECs.")
nx_visu_LP(cities, True, x, [], 9999)

# if x variables were originally continuous, the final integer solution may violate SECs!
# if you wish to avoid this, set x variables to binary right from the beginning (see lecture slides)
