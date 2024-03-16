#!/usr/bin/python

n = 3  # number of orders
L = 11  # length of (raw) rolls

# demands
d = [2, 5, 3]
# lengths
l = [7, 3, 4]

m = 10  # number of available (raw) rolls (=known upper bound on the optimum)


import cscolgen

cscolgen.solve(m, L, d, l)


"""
import cuttingstockmodel
cuttingstockmodel.solve(m, L, d, l)

import csflowmodel
csflowmodel.solve(m, L, d, l)
"""
