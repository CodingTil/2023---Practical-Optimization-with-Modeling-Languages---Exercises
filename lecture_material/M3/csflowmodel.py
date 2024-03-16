from gurobipy import Model, GRB, quicksum

# cutting stock model


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

    # create the graph
    # vertices 0, 1, ..., L represent the L units of (used) length of raw roll
    # at each state (vertex v_j) we can decide to
    #   - cut order i using an arc A_1^i = {(v_k, v_{k+l_i}) | 0 <= k <= L-l_i}
    #   - waste the next unit using an arc in A_2 = {(v_k, v_{k+1}) | 0 <= k <= L-1}
    # start node -1 with single edge (-1, 0)
    A1 = dict()
    for i in range(len(d)):
        A1[i] = [(j, j + l[i]) for j in range(0, L - l[i] + 1)]
    A2 = [(j, j + 1) for j in range(-1, L)]
    arcs = A2
    for i in range(len(d)):
        arcs += A1[i]

    model = Model("Cutting Stock")

    # create variables
    x = {}
    y = {}
    for i in range(len(d)):
        for arc in A1[i]:
            x[i, arc] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}_{arc}", lb=0)
    for arc in A2:
        if arc[0] >= 0:
            type = GRB.CONTINUOUS
        else:
            type = GRB.INTEGER
        y[arc] = model.addVar(vtype=type, name=f"y_{arc}")

    # set objective
    model.setObjective(y[(-1, 0)], GRB.MINIMIZE)

    # add constraints
    for j in range(0, L):
        model.addConstr(
            quicksum(x[i, arc] for i in range(len(d)) for arc in A1[i] if arc[1] == j)
            + quicksum(y[arc] for arc in A2 if arc[1] == j)
            == quicksum(
                x[i, arc] for i in range(len(d)) for arc in A1[i] if arc[0] == j
            )
            + quicksum(y[arc] for arc in A2 if arc[0] == j)
        )

    for i in range(len(d)):
        model.addConstr(quicksum(x[i, arc] for arc in A1[i]) >= d[i])

    # solve model
    model.relax()
    model.optimize()

    # print solution
    print(f"Optimal value: {model.objVal}")
