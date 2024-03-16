from gurobipy import Model, GRB, quicksum, tuplelist


# number n nodes (indices 0, ..., n-1)
# E weighted directed edges (i, j, w) with i, j in {0, ..., n-1} and w >= 0
# s source node (0 <= s < n)
# t target node (0 <= t < n)
def solve(n, E, s, t):
    model = Model("shortest path")

    seen_nodes = set()
    x = {}
    for e in E:
        # x[(e[0],e[1])] = model.addVar(vtype=GRB.BINARY, name="x_"+str(e[0])+"_"+str(e[1]))
        x[(e[0], e[1])] = model.addVar(
            vtype=GRB.CONTINUOUS, name="x_" + str(e[0]) + "_" + str(e[1])
        )
        seen_nodes.add(e[0])
        seen_nodes.add(e[1])

    for i in seen_nodes:
        from_i = get_edge_from(i, E)
        to_i = get_edge_to(i, E)
        if i == s:
            model.addConstr(
                quicksum(x[(i, e[1])] for e in from_i)
                - quicksum(x[(e[0], i)] for e in to_i)
                == 1
            )
        elif i == t:
            model.addConstr(
                quicksum(x[(e[0], i)] for e in to_i)
                - quicksum(x[(i, e[1])] for e in from_i)
                == 1
            )
        else:
            model.addConstr(
                quicksum(x[(i, e[1])] for e in from_i)
                == quicksum(x[(e[0], i)] for e in to_i)
            )

    model.setObjective(quicksum(x[(e[0], e[1])] * e[2] for e in E), GRB.MINIMIZE)
    model.optimize()

    print(f"Optimal solution: {model.objVal}")


def get_edge_from(i, E):
    return list(e for e in E if e[0] == i)


def get_edge_to(i, E):
    return list(e for e in E if e[1] == i)
