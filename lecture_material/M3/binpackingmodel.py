from gurobipy import Model, GRB, quicksum


# m number of bins
# a list of item sizes
# b bin capacity
def solve(m, a, b):
    model = Model("Bin Packing")

    # create variables
    x = {}
    for i in range(len(a)):
        for j in range(m):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    y = {}
    for j in range(m):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")

    # set objective
    model.setObjective(quicksum(y[j] for j in range(m)), GRB.MINIMIZE)

    # add constraints
    for i in range(len(a)):
        model.addConstr(quicksum(x[i, j] for j in range(m)) == 1)
    for j in range(m):
        model.addConstr(quicksum(a[i] * x[i, j] for i in range(len(a))) <= b * y[j])

    for i in range(len(a)):
        for j in range(m):
            model.addConstr(x[i, j] <= y[j])

    # break symmetry of y: y[0] >= y[1] >= ... >= y[m-1]
    for j in range(m - 1):
        model.addConstr(y[j] >= y[j + 1])

    # solve model
    model.optimize()

    # print solution
    print(f"Optimal value: {model.objVal}")
