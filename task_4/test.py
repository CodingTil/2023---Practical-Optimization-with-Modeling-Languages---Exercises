import random
from typing import List, Callable
import tempfile
import math

import networkx as nx
from gurobipy import Model, GRB, quicksum


def generate_random_graphs(
    n: int = 10,
    probability_generator: Callable[[], float] = lambda: math.sqrt(
        random.uniform(0, 1)
    ),
) -> List[nx.Graph]:
    """
    Generates n different random graphs with n nodes each.

    Parameters
    ----------
    n : int, optional
        The number of graphs to generate. The number of nodes in each graph is equal to n. The default is 10.
    probability_generator : Callable[[], float], optional
        A function that returns a probability (float between 0 and 1) for the creation of an edge in the random graph. The default is lambda: math.sqrt(random.uniform(0, 1)) (because more dense graphs are more interesting to look at)

    Returns
    -------
    List[nx.Graph]
        A list of n different random graphs with n nodes each.
    """
    # Initialize an empty list to store the graphs
    graphs: list[nx.Graph] = []

    # For each number i from 1 to n
    for i in range(1, n + 1):
        # Generate i different random graphs of size i
        for _ in range(i):
            # Create a random graph with i nodes. The probability of an edge being created is set to a random value
            G = nx.erdos_renyi_graph(i, probability_generator())
            while G in graphs:
                G = nx.erdos_renyi_graph(i, probability_generator())
            # Add the graph to the list
            graphs.append(G)

    return graphs


def normal_vertex_coloring(G: nx.Graph) -> Model:
    """
    Solves the vertex coloring problem for the given graph G.

    Parameters
    ----------
    G : nx.Graph
        The graph to solve the vertex coloring problem for.

    Returns
    -------
    Model
        The solved gurobipy model instance.
    """
    model = Model()

    # use basic model (see https://arxiv.org/pdf/1706.10191.pdf 2.2)

    x = {}
    y = {}
    for c in range(0, G.number_of_nodes()):
        y[c] = model.addVar(vtype=GRB.BINARY, name="y_" + str(c))
        for v in G.nodes:
            x[v, c] = model.addVar(vtype=GRB.BINARY, name="x_" + str(v) + "_" + str(c))

    model.update()

    model.setObjective(
        quicksum(y[c] for c in range(0, G.number_of_nodes())), GRB.MINIMIZE
    )

    # each vertex is colored
    for v in G.nodes:
        model.addConstr(quicksum(x[v, c] for c in range(0, G.number_of_nodes())) == 1)

    # adjacent vertices have different colors
    for (u, v) in G.edges:
        for c in range(0, G.number_of_nodes()):
            model.addConstr(x[u, c] + x[v, c] <= y[c])

    for c in range(0, G.number_of_nodes()):
        for v in G.nodes:
            model.addConstr(x[v, c] <= y[c])

    # from here: symmetry breaking constraints
    for c in range(1, G.number_of_nodes()):
        model.addConstr(y[c - 1] >= y[c])

    for c in range(0, G.number_of_nodes()):
        model.addConstr(y[c] <= quicksum(x[v, c] for v in G.nodes))

    model.optimize()

    return model


def write_adjlist_to_tempfile(G: nx.Graph):
    """
    Writes the adjacency list of the given graph G to a temporary file.

    Parameters
    ----------
    G : nx.Graph
        The graph to write the adjacency list of.

    Returns
    -------
    tempfile.NamedTemporaryFile
        The temporary file containing the adjacency list of the given graph G.
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=True)

    # Write the adjacency list of the graph to the file
    nx.write_adjlist(G, temp_file.name)

    return temp_file


from vc_bc import solve
from vc_bc_cut_check import run_test


MAX_GRAPH_SIZE = 25


iteration = 0
for G in generate_random_graphs(MAX_GRAPH_SIZE):
    # generate the temporary file containing the adjacency list of G
    tmp_file = write_adjlist_to_tempfile(G)

    test_model = normal_vertex_coloring(G)
    comparison_model, _ = solve(tmp_file.name, True)

    # both models should come to the same result (total number of colors)
    if abs(test_model.objVal - comparison_model.objVal) > 0.0001:
        print("Test failed for graph", G.nodes)
        print("Test model:", test_model.objVal)
        print("Comparison model:", comparison_model.objVal)
        raise Exception("Test failed")

    # check if the solution is valid (i.e. the cuts are correct)
    if not run_test(tmp_file.name):
        print("Test failed for graph", G.nodes)
        raise Exception("Test failed")

    # delete the temporary file
    tmp_file.close()

    iteration += 1
    print("Test", iteration, "passed. Graph size:", G.number_of_nodes())
