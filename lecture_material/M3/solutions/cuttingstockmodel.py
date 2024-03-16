#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2018, modified 2019
Last modified on Fri May 15 15:08:27 2020

@author: bastubbe, luebbecke
"""


from gurobipy import *


def solve(m, L, d, l):
    # number of orders
    n = len(d)

    # aggregate data if we have several orders of identical length
    agg = {}  # dictionary with length : demand pairs
    for i in range(n):
        if l[i] not in agg:
            agg[l[i]] = d[i]
            continue
        agg[l[i]] += d[i]

    # now update the input data
    l = list(agg.keys())
    d = list(agg.values())
    n = len(d)

    # model
    model = Model("cuttingstock")

    # create decision variables

    # how often to cut an order from a roll
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(name="x_" + str(i) + "_" + str(j), vtype=GRB.INTEGER)

    # use a roll?
    y = {}
    for j in range(m):
        y[j] = model.addVar(name="y_" + str(j), obj=1, vtype=GRB.BINARY)

    # capacity/length of a raw roll cannot be exceeded
    for j in range(m):
        model.addConstr(
            quicksum(l[i] * x[i, j] for i in range(n)) <= L, "capacity_" + str(j)
        )

    # fulfill all orders
    for i in range(n):
        model.addConstr(
            quicksum(x[i, j] for j in range(m)) == d[i], name="demand_" + str(i)
        )

    # linking constraints between x and y
    for i in range(n):
        for j in range(m):
            model.addConstr(
                x[i, j] <= d[i] * y[j], name="link_" + str(i) + "_" + str(j)
            )

    # as always for debugging
    model.write("cscompact.lp")

    # off we go
    model.optimize()
