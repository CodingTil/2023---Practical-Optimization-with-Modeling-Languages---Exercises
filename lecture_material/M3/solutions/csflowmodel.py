#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:31:01 2018
Last modified 2020

@author: luebbecke
"""


from gurobipy import *

# this is a flow based model for cutting stock. A cutting pattern is represented as a path in a graph
# that contains nodes for each capacity unit of the raw roll.


def solve(m, L, d, l):
    # number of orders
    n = len(d)

    # "colored" edges, representing cutting orders
    A1 = []
    for i in range(n):
        A1temp = []
        for v in range(0, L - l[i] + 1):
            A1temp.append((v, v + l[i]))
        A1.append(A1temp)

    # "black" edges, representing "wasted" capacity
    A2 = []
    for v in range(-1, L):  # node "-1" is the start node
        A2.append((v, v + 1))

    model = Model("CS flow model")

    x1 = {}
    for i in range(n):
        for e in A1[i]:
            x1[i, e] = model.addVar(
                name="x1_" + str(i) + "_" + str(e), vtype=GRB.CONTINUOUS, obj=0.0
            )

    x2 = {}
    for e in A2:
        if e[0] == -1:
            objcoeff = 1.0
        else:
            objcoeff = 0.0
        x2[e] = model.addVar(name="x2_" + str(e), vtype=GRB.INTEGER, obj=objcoeff)

    # flow conservation in node v
    for v in range(L):
        model.addConstr(  # flow into v
            quicksum(x1[i, e] for i in range(n) for e in A1[i] if e[1] == v)
            + quicksum(x2[e] for e in A2 if e[1] == v)
            -
            # flow out of v
            quicksum(x1[i, e] for i in range(n) for e in A1[i] if e[0] == v)
            - quicksum(x2[e] for e in A2 if e[0] == v)
            == 0
        )

    # satisfy demands
    for i in range(n):
        model.addConstr(quicksum(x1[i, e] for e in A1[i]) >= d[i])

    model.write("csflow.lp")
    model.optimize()

    # solve LP relaxation


#    r = model.relax()
#    r.optimize()
