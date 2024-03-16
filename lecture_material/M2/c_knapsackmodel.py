from gurobipy import Model, GRB, quicksum


# item sizes a, item profits p, capacity b
def solve(a, p, b, C):
    model = Model("knapsack")

    # a binary variable per item (selected or not); gives profit if selected
    x = dict()
    for i in range(len(a)):
        x[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        # x[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}")

    # capacity constraint
    model.addConstr(
        quicksum(a[i] * x[i] for i in range(len(a))) <= b,
    )

    # respect conflicts
    for i, j in C:
        model.addConstr(x[i] + x[j] <= 1)

    # optimize
    model.setObjective(quicksum(p[i] * x[i] for i in range(len(a))), GRB.MAXIMIZE)
    model.optimize()

    # print solution
    print(f"Optimal solution: {model.objVal}")
