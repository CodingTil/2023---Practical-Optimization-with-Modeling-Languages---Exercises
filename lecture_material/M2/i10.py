# item sizes
a = [7, 4, 6, 4, 5, 4, 3, 4, 6, 7]

# profits
p = [5, 4, 4, 6, 4, 7, 4, 5, 7, 3]

# knapsack capacity
b = 20

# import model and solve
import knapsackmodel

knapsackmodel.solve(a, p, b)

# translate problem into shortest path problem and solve
import shortestpath

capacity_units = list(range(0, b + 1))
item_indices = list(range(0, len(a) + 1))

nodes = list((i, j) for i in capacity_units for j in item_indices)

item_arcs = list(
    ((c, i - 1), (c + a[i - 1], i), -p[i - 1])
    for i in range(1, len(a) + 1)
    for c in range(0, b - a[i - 1] + 1)
)
skip_arcs = list(
    ((c, i), (c, i + 1), 0) for i in range(0, len(a)) for c in range(0, b + 1)
)
waste_arcs = list(
    ((c, i), (c + 1, i), 0) for i in range(0, len(a) + 1) for c in range(0, b)
)

arcs = item_arcs + skip_arcs + waste_arcs

shortestpath.solve(len(nodes), arcs, (0, 0), (b, len(a)))
