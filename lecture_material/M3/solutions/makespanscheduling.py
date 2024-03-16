#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 00:31:00 2020

@author: luebbecke
"""

from gurobipy import *


def solve(m, a, b):
    # number of jobs
    n = len(a)

    model = Model("makespan scheduling")

    # assign job i to machine j
    x = {}
    for i in range(n):
        x[i] = {}
        for j in range(m):
            x[i][j] = model.addVar(vtype="b")

    # makespan
    cmax = model.addVar(obj=1)

    # every job i must be performed on exactly one machine j
    for i in range(n):
        model.addConstr(quicksum(x[i][j] for j in range(m)) == 1)

    # total assigned duration to machine j is a lower bound on the makespan
    for j in range(m):
        model.addConstr(quicksum(a[i] * x[i][j] for i in range(n)) <= cmax)

    # objective cut
    # model.addConstr(cmax <= 70)

    model.write("makespan.lp")
    model.optimize()
