#!/usr/bin/python


from gurobipy import *


n = 10  # number of orders
L = 20  # length of (raw) rolls

# demands
d = [5, 4, 4, 6, 4, 3, 2, 2, 3, 3]

# lengths
l = [7, 4, 6, 1, 5, 2, 3, 8, 9, 10]

# number of available (raw) rolls (=known upper bound on the optimum)
m = 36


import cuttingstockmodel

# cuttingstockmodel.solve(m, L, d, l)


import csflowmodel

csflowmodel.solve(m, L, d, l)
