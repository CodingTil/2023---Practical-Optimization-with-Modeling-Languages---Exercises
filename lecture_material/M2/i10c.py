# item sizes
a = [7, 4, 6, 4, 5, 4, 3, 4, 6, 7]

# profits
p = [5, 4, 4, 6, 4, 7, 4, 5, 7, 3]

# knapsack capacity
b = 20

# conflicts; item indices start a 0
C = [
    (0, 4),
    (0, 7),
    (0, 8),
    (0, 9),
    (1, 4),
    (1, 7),
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (3, 8),
    (4, 5),
    (4, 6),
    (4, 8),
    (5, 7),
    (5, 8),
    (6, 8),
    (7, 9),
    (8, 9),
]


# import model and solve
import c_knapsackmodel

c_knapsackmodel.solve(a, p, b, C)
