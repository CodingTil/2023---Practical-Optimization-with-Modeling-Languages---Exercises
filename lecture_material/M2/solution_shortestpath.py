"""
Created on Wed Apr 29 00:11:39 2020

@author: luebbecke
"""

from gurobipy import Model, GRB, quicksum, tuplelist


def solve(n, E, s, t):
    model = Model("shortest path")

    # a (binary) variable per edge
    x = {}
    for e in E:
        x[(e[0], e[1])] = model.addVar(
            vtype="b", obj=e[2], name="x_" + str(e[0]) + "_" + str(e[1])
        )

    # out of curiosity, try gruobi's tuplelist
    T = tuplelist(E)

    # flow conservation constraints
    for i in range(n):
        # right hand side
        rhs = 0
        if i == s:
            rhs = 1
        if i == t:
            rhs = -1

        # now the flow balance
        model.addConstr(
            # outgoing edges
            quicksum(x[(i, j)] for j in list(t[1] for t in T.select(i, "*"))) -
            # incoming edges
            quicksum(x[(j, i)] for j in list(t[0] for t in T.select("*", i))) == rhs,
            name="flow_" + str(i),
        )

    # model.write("model.lp")
    model.optimize()
