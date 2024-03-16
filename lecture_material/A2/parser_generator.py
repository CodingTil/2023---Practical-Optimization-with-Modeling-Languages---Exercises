from typing import Dict, Tuple, Callable
import math

import networkx as nx
from networkx import Graph, DiGraph
from gurobipy import Model, GRB, quicksum


def parse_tsp_file(filename: str) -> Dict[int, Tuple[float, float]]:
    file_content = open(filename, "r").read()  # read the file content

    # Get n from line "DIMENSION: n"
    n = int(
        next(
            filter(lambda x: x.startswith("DIMENSION"), file_content.split("\n"))
        ).split(":")[1]
    )

    coordinates: Dict[int, Tuple[float, float]] = {}
    for i in range(1, n + 1):
        # Get line "i x_i y_i" and save "(x_i, y_i)" to a list
        coordinates[i] = tuple(
            map(
                float,
                next(
                    filter(lambda x: x.startswith(str(i)), file_content.split("\n"))
                ).split(" ")[1:3],
            )
        )

    return coordinates


def coordinates_to_directed_graph(
    coordinates: Dict[int, Tuple[float, float]]
) -> DiGraph:
    graph = DiGraph()

    for i in coordinates:
        graph.add_node(i, pos=coordinates[i])

    for i in coordinates:
        for j in coordinates:
            if i != j:
                graph.add_edge(i, j, weight=euclid_2d(coordinates[i], coordinates[j]))
                graph.add_edge(j, i, weight=euclid_2d(coordinates[i], coordinates[j]))

    return graph


def euclid_2d(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    return max(
        0, round(math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2))
    )


def create_tsp_mtz_model(graph: DiGraph) -> Model:
    model = Model("TSP")

    # Create variables
    x = {}
    for edge in graph.edges:
        x[edge] = model.addVar(vtype=GRB.BINARY, name=f"x_{edge[0]}_{edge[1]}")

    u = {}
    for i in graph.nodes:
        u[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"u_{i}")

    model.update()

    # Set objective
    model.setObjective(
        quicksum(graph.edges[edge]["weight"] * x[edge] for edge in graph.edges),
        sense=GRB.MINIMIZE,
    )

    model.update()

    # Add constraints
    for i in graph.nodes:
        model.addConstr(quicksum(x[edge] for edge in graph.in_edges(i)) == 1)
        model.addConstr(quicksum(x[edge] for edge in graph.out_edges(i)) == 1)

    model.addConstr(u[1] == 1)

    for edge in graph.edges:
        if edge[1] != 1:
            model.addConstr(
                u[edge[0]] - u[edge[1]] + len(graph.nodes) * x[edge]
                <= len(graph.nodes) - 1
            )

    model.update()

    return model


def create_tsp_dfj_model(digraph: DiGraph) -> Tuple[Model, Callable]:
    graph: Graph = digraph.to_undirected()

    model = Model("TSP")

    # Create variables
    x = {}
    for edge in graph.edges:
        x[edge] = model.addVar(vtype=GRB.BINARY, name=f"x_{edge[0]}_{edge[1]}")

    model.update()

    # Set objective
    model.setObjective(
        quicksum(graph.edges[edge]["weight"] * x[edge] for edge in graph.edges),
        sense=GRB.MINIMIZE,
    )

    model.update()

    # Add constraints
    for i in graph.nodes:
        model.addConstr(
            quicksum(x[edge] for edge in graph.edges if edge[0] == i or edge[1] == i)
            == 2
        )

    model.update()

    # add callback for sub-tour elimination
    def subtourelim(model, where):
        if not (
            where == GRB.Callback.MIPSOL
            or (
                where == GRB.Callback.MIPNODE
                and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL
            )
        ):
            return

        # get the values of the x variables
        if where == GRB.Callback.MIPSOL:
            x_values = dict(model.cbGetSolution(x))
        else:
            x_values = dict(model.cbGetNodeRel(x))

        # create a new graph with the x values as edge weights
        g_cut = Graph()
        for edge in x_values:
            g_cut.add_edge(edge[0], edge[1], weight=max(0, x_values[edge]))

        # compute capacity of minimum cut
        capacity, partition = nx.stoer_wagner(g_cut)

        EPSILON = 10 ** (-5)

        if capacity < 2 - EPSILON:
            edge_list = []
            for p_1 in partition[0]:
                for p_2 in partition[1]:
                    if (p_1, p_2) in x_values:
                        edge_list.append(x[p_1, p_2])
                    elif (p_2, p_1) in x_values:
                        edge_list.append(x[p_2, p_1])

            model.cbLazy(quicksum(edge_list) >= 2)

    model.params.LazyConstraints = 1

    return model, subtourelim
