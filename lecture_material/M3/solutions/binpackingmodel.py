#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 00:31:00 2020

@author: luebbecke
"""

from gurobipy import *


def solve(m, a, b):
    # number of items
    n = len(a)

    model = Model("binpacking")

    # assign item i to bin j
    x = {}
    for i in range(n):
        x[i] = {}
        for j in range(m):
            x[i][j] = model.addVar(vtype="b")

    # open bin j
    y = {}
    for j in range(m):
        y[j] = model.addVar(vtype="b", obj=1)

    # pack every item in exactly one bin
    for i in range(n):
        model.addConstr(quicksum(x[i][j] for j in range(m)) == 1)

    # respect resource capacities in each bin
    for j in range(m):
        model.addConstr(quicksum(a[i] * x[i][j] for i in range(n)) <= b * y[j])

    """
    # additional linking constraint
    for i in range(n):
        for j in range(m):
            model.addConstr(x[i][j] <= y[j])

    """

    model.write("bpp.lp")
    model.optimize()
