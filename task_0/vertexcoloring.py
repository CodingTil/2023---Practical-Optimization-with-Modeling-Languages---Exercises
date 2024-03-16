import networkx as nx
from gurobipy import *


def solve(instance_path):
    # Read Graph from instance path
    G = nx.read_adjlist(instance_path)

    # Model
    model = Model("Vertex coloring")
    model.modelSense = GRB.MINIMIZE

    # number of colors is given by number of nodes
    colors = range(len(G.nodes))

    # Decision variable x_(v,c) indicates whether vertex v has color c (value 1) or
    # not (value 0)
    # TODO: Adjust f-string formatting so that the variables in the gurobi model
    # will have the correct name! Also use reasonable dict keys for x[...] and y[...]
    x = {}
    for vertex in G.nodes:
        for color in colors:
            x[vertex, color] = model.addVar(
                name=f"x_{vertex}_{color}", vtype=GRB.BINARY
            )

    y = {}
    for color in colors:
        y[color] = model.addVar(name=f"y_{color}", vtype=GRB.BINARY)
    # -------------------------------------------------------------------------

    # Update the model to make variables known.
    # From now on, no variables should be added.
    model.update()

    # TODO: Add your constraints ----------------------------------------------

    # Each vertex has to be assigned exactly one color
    for vertex in G.nodes:
        model.addConstr(quicksum(x[vertex, color] for color in colors) == 1)

    # No two adjacent vertices can have the same color
    for edge in G.edges:
        v1, v2 = edge
        for color in colors:
            model.addConstr(x[v1, color] + x[v2, color] <= 1)

    # If a color is assigned, then its corresponding y-variable has to be 1
    for color in colors:
        model.addConstr(
            quicksum(x[vertex, color] for vertex in G.nodes) <= len(G.nodes) * y[color]
        )

    # -------------------------------------------------------------------------

    model.update()
    # For debugging: print your model
    # model.write('model.lp')

    # Set objective
    model.setObjective(quicksum(y[color] for color in colors), GRB.MINIMIZE)

    model.update()

    # Optimize model
    model.optimize()

    # TODO: Adjust your dict keys for y[...] to print out the selected --------
    # colors from your solution and then uncomment those lines.
    # This is not not necessary for grading but helps you confirm that your
    # model works

    # Printing solution and objective value
    def printSolution():
        if model.status == GRB.OPTIMAL:
            print("\n objective: %g\n" % model.ObjVal)
            print("Selected following colors:")
            for color in colors:
                pass
                # if y["TODO"].x >= 0.9: # for numerical reasons never compare against 1
                # vertices_string = ""
                # for vertex in G.nodes:
                #     if x["TODO"].x >= 0.9:
                #         vertices_string += f"{vertex} "
                # print(f"Color {color} is assigned to vertices {vertices_string}")
        else:
            print("No solution!")

    # -------------------------------------------------------------------------

    printSolution()
    # Please do not delete the following line
    return model


if __name__ == "__main__":
    solve(instance_path="vertexcoloring_data1.adjlist")
