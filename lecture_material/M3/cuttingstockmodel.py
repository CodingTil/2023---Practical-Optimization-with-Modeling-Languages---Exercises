from gurobipy import Model, GRB, quicksum


# m number of available (raw) rolls
# L length of (raw) rolls
# d demands
# l lengths
def solve(m, L, d, l):
    # preprocess: if two orders i!=j have identical length l[i]=l[j], then we can replace those by a single order with demand d[i]+d[j] and length l[i]
    # for i in range(len(d)):
    #     for j in range(i+1, len(d)):
    #         if l[i]==l[j]:
    #             d[i] += d[j]
    #             d[j] = 0
    # d = [d[i] for i in range(len(d)) if d[i]>0]

    model = Model("Cutting Stock")

    # create variables
    x = {}
    for i in range(len(d)):
        for j in range(m):
            x[i, j] = model.addVar(vtype=GRB.INTEGER, name=f"x_{i}_{j}")
    y = {}
    for j in range(m):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")

    # set objective
    model.setObjective(quicksum(y[j] for j in range(m)), GRB.MINIMIZE)

    # add constraints
    for i in range(len(d)):
        model.addConstr(quicksum(x[i, j] for j in range(m)) == d[i])
    for j in range(m):
        model.addConstr(quicksum(l[i] * x[i, j] for i in range(len(d))) <= L)
    for i in range(len(d)):
        for j in range(m):
            model.addConstr(x[i, j] <= d[i] * y[j])

    # break symmetry of y: y[0] >= y[1] >= ... >= y[m-1]
    for j in range(m - 1):
        model.addConstr(y[j] >= y[j + 1])

    # solve model
    model.optimize()

    # print solution
    print(f"Optimal value: {model.objVal}")
