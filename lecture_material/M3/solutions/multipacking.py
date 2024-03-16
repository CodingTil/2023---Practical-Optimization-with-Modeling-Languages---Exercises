#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:18:18 2020

@author: luebbecke
"""

from gurobipy import *


def solve(I, B, R, required, available, copies):
    model = Model("multi-dimensional multi-packing")

    # assignment of items to bins ("placement")
    x = {}
    for i in range(I):
        x[i] = {}
        for b in range(B):
            x[i][b] = model.addVar(vtype="b")

    # decide whether bin should be opened
    # these variables are *not* necessary, all bins are open anyway!
    y = {}
    for b in range(B):
        # if number of bins should be minimized simply change the objective coefficient
        y[b] = model.addVar(vtype="b", obj=0)

    # surplus variables to measure the load excess in a bin
    s = {}
    for b in range(B):
        s[b] = model.addVar(obj=1)

    # ensure that correct number of copies of each item are placed
    for i in range(I):
        model.addConstr(quicksum(x[i][b] for b in range(B)) == copies[i])

    # respect resource capacities in each bin
    for r in range(R):
        for b in range(B):
            model.addConstr(
                quicksum(required[i][r] * x[i][b] for i in range(I))
                <= available[b][r] * y[b]
            )

    # measure deviation from average load in each bin
    for b in range(B):
        model.addConstr(
            quicksum(x[i][b] for i in range(I)) - sum(copies[i] for i in range(I)) / B
            <= s[b]
        )

    model.write("multietc.lp")
    model.optimize()
