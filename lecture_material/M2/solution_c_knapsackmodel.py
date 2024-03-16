"""
Created on Tue Apr 28 00:31:05 2020

@author: luebbecke
"""

from gurobipy import Model, GRB, quicksum


# item sizes a, item profits p, capacity b, conflicts C
def solve(a, p, b, C):
    model = Model("knapsack with conflicts")

    # a binary variable per item (selected or not); gives profit if selected
    x = {}
    for i in range(len(a)):
        x[i] = model.addVar(vtype="b", obj=p[i])

    # capacity constraint
    model.addConstr(quicksum(a[i] * x[i] for i in range(len(a))) <= b)

    # respect conflicts
    for i, j in C:
        model.addConstr(x[i] + x[j] <= 1)

    # by default gurobi is minimizing
    model.ModelSense = GRB.MAXIMIZE

    # write model to file, helps in debugging
    model.write("model.lp")

    # off we go
    model.optimize()

    # trivial output
    print("we select the following items:")
    for i in range(len(a)):
        if x[i].x > 0.1:
            print(i)
    print("the profit of this packing is " + str(model.objval))
