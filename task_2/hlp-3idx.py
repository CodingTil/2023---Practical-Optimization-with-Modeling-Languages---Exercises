from gurobipy import *
import json


# You do not need to modify this function
def read_instance(instance_path):
    with open(instance_path) as f:
        data = json.load(f)
    return (
        data["p"],
        data["c"],
        data["alpha"],
        data["customers"],
        data["distances"],
        data["demands"],
    )


def solve(p, c, alpha, customers, distances, demands):
    """Solves the hub location problem using the 3-index formulation.

    Args:
        p (int): number of hubs to open
        c (float): cost factor for flows between customer and hub in money units per demand and distance unit
        alpha (float): discount factor for intra-hub transports
        customers (list[int]): list of customer indices
        distances (list[list[float]]): distance matrix, distances[i][j] is the distance between customer i and customer j
        demands (list[list[float]]): demand matrix, demands[i][j] is the demand from customer i to customer j

    Returns:
        Gurobi.Model: the solved model
    """
    model = Model("Hub Location")

    # You can try and disable cutting planes - what happens?
    # model.setParam(GRB.Param.Cuts, 0)
    # NOTE: Please do not turn off cutting planes in your final submission

    # --- Do we need to prepare something? ---

    # --- Variables ---
    # y_i_k_l: amount of demand sent from customer i via hub k to hub l
    y = {}
    for i in customers:
        for k in customers:
            for l in customers:
                if k != l:
                    y[i, k, l] = model.addVar(vtype="c", lb=0, name=f"y_{i}_{k}_{l}")

    # more variables are necessary
    # z_i_k: if customer i is assigned to hub k
    z = {}
    for i in customers:
        for k in customers:
            z[i, k] = model.addVar(vtype="b", name=f"z_{i}_{k}")

    model.update()

    # --- Constraints ---
    # open at most p hubs
    model.addConstr(quicksum(z[k, k] for k in customers) <= p)

    # assign customer i to hub k if it is open
    for i in customers:
        for k in customers:
            model.addConstr(z[i, k] <= z[k, k])

    # single assignment: one customer one hub
    for i in customers:
        model.addConstr(quicksum(z[i, k] for k in customers) == 1)

    # intra-hub flows only allowed if hub is open
    for i in customers:
        for k in customers:
            for l in customers:
                if k != l:
                    model.addConstr(
                        y[i, k, l]
                        <= z[k, k] * quicksum(demands[i][j] for j in customers)
                    )
                    model.addConstr(
                        y[i, k, l]
                        <= z[l, l] * quicksum(demands[i][j] for j in customers)
                    )

    # flow conservation
    for i in customers:
        for k in customers:
            model.addConstr(
                # incoming
                quicksum(y[i, l, k] for l in customers if l != k)
                + quicksum(demands[i][j] for j in customers) * z[i, k]
                ==
                # outgoing
                quicksum(y[i, k, l] for l in customers if l != k)
                + quicksum(demands[i][j] * z[j, k] for j in customers)
            )

    model.setObjective(
        quicksum(
            (demands[i][j] + demands[j][i]) * distances[i][k] * z[i, k] * c
            for k in customers
            for i in customers
            for j in customers
        )
        + quicksum(
            (y[i, k, l] * distances[k][l] * alpha * c)
            for i in customers
            for k in customers
            for l in customers
            if l != k
        ),
        GRB.MINIMIZE,
    )

    # --- Solve model ---

    # If you want to solve just the LP relaxation, uncomment the lines below
    # model.update()
    # model = model.relax()

    model.optimize()
    # model.write("model.lp")

    # If your model is infeasible (but you expect it to not be), comment out the lines below to compute and write out a infeasible subsystem (Might take very long)
    # model.computeIIS()
    # model.write("model.ilp")

    return model


if __name__ == "__main__":
    p, c, alpha, customers, distances, demands = read_instance("n50.json")

    solve(p, c, alpha, customers, distances, demands)
