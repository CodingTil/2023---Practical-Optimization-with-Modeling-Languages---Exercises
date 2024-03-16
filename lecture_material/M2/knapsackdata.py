"""
Created on Mon Nov 25 16:56:56 2019

@author: luebbecke
"""

# item sizes
a = [4, 17, 10, 9, 6, 7, 8]
# profits
p = [10, 14, 3, 16, 7, 3, 6]
# knapsack capacity
b = 27

# import model and solve
import knapsackmodel

knapsackmodel.solve(a, p, b)
