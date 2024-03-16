#!/usr/bin/python


from gurobipy import *


n = 20  # number of orders
L = 40  # length of (raw) rolls

# demands
d = [5, 7, 1, 6, 2, 5, 4, 3, 6, 2, 1, 7, 7, 1, 5, 5, 2, 3, 7, 3]

# lengths
l = [5, 6, 6, 9, 5, 9, 7, 8, 8, 6, 5, 9, 5, 9, 6, 8, 9, 9, 7, 7]

# number of available (raw) rolls (=known upper bound on the optimum)
m = 82


import cscolgen

cscolgen.solve(m, L, d, l)


"""
import cuttingstockmodel
cuttingstockmodel.solve(m, L, d, l)

import csflowmodel
csflowmodel.solve(m, L, d, l)
"""
