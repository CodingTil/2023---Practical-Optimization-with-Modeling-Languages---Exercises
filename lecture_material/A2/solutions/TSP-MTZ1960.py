# -*- coding: utf-8 -*-
"""
solve a TSPLIB input file via Miller, Tucker, Zemlin (1960)

@author: luebbecke
"""


import sys
import re
import math
from gurobipy import *

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


###############################################################################

# visualization of final solution
def visu_LP(cities, x):
    plt.plot(cities[0], cities[1], "o")
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if x[i, j].x > 0.0001:
                plt.plot(
                    [cities[0][i], cities[0][j]],
                    [cities[1][i], cities[1][j]],
                    color="black",
                    alpha=x[i, j].x,
                    linewidth=x[i, j].x,
                )
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
    # this line must constain the coordinates of a city
    c = int(line.split()[0])
    xcoord = float(line.split()[1])
    ycoord = float(line.split()[2])
    cities[0].append(xcoord)
    cities[1].append(ycoord)

nodes = range(n)

###############################################################################

model = Model("TSP-MTZ")

# "sequence" of cities
u = {}
for i in nodes:
    u[i] = model.addVar(name="u_" + str(i), obj=0.0)

# directed graph; edge in tour?
x = {}
for i in nodes:
    for j in nodes:
        if i == j:
            continue
        x[i, j] = model.addVar(
            name="x_" + str(i) + "_" + str(j),
            vtype="b",
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
    model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == 1)
    model.addConstr(quicksum(x[j, i] for j in nodes if i != j) == 1)

# subtour elimination
for i in nodes:
    for j in nodes:
        if i == j or j == 0:
            continue
        model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

model.write("TSPMTZ.lp")

model.optimize()


# visualization of optimal tour
visu_LP(cities, x)
