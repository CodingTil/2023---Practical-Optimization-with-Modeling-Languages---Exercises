from gurobipy import Model, GRB, quicksum


# m number of bins
# a list of item sizes
# b bin capacity
def solve(m, a, b, upperbound=None):
    model = Model("Makespan Scheduling")

    # create variables
    x = {}
    for i in range(len(a)):
        for j in range(m):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    C_max = model.addVar(vtype=GRB.INTEGER, name="C_max")

    # set objective
    model.setObjective(C_max, GRB.MINIMIZE)

    # add constraints
    for i in range(len(a)):
        model.addConstr(quicksum(x[i, j] for j in range(m)) == 1)
    for j in range(m):
        model.addConstr(quicksum(a[i] * x[i, j] for i in range(len(a))) <= C_max)

    if upperbound is not None:
        model.addConstr(C_max <= upperbound)

    # solve model
    model.optimize()

    # print solution
    print(f"Optimal value: {model.objVal}")
