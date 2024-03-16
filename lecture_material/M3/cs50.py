#!/usr/bin/python


from gurobipy import *


n = 50  # number of orders
L = 60  # length of (raw) rolls

# demands
d = [
    2,
    4,
    3,
    2,
    3,
    3,
    6,
    6,
    6,
    5,
    7,
    2,
    1,
    5,
    4,
    7,
    1,
    6,
    1,
    4,
    4,
    7,
    2,
    2,
    5,
    4,
    3,
    4,
    3,
    4,
    5,
    4,
    5,
    5,
    5,
    1,
    5,
    2,
    4,
    2,
    6,
    3,
    3,
    4,
    7,
    4,
    3,
    5,
    2,
    1,
]

# lengths
l = [
    5,
    6,
    5,
    7,
    9,
    6,
    8,
    7,
    5,
    6,
    5,
    9,
    5,
    8,
    7,
    7,
    9,
    7,
    6,
    6,
    6,
    8,
    5,
    5,
    5,
    9,
    6,
    7,
    9,
    5,
    6,
    9,
    6,
    6,
    9,
    8,
    7,
    7,
    5,
    9,
    8,
    7,
    8,
    6,
    5,
    7,
    5,
    5,
    7,
    6,
]

# number of available (raw) rolls (=known upper bound on the optimum)
m = 192


import cuttingstockmodel

# cuttingstockmodel.solve(m, L, d, l)


import csflowmodel

csflowmodel.solve(m, L, d, l)
