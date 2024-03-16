import sys

import networkx as nx
from parser_generator import (
    parse_tsp_file,
    coordinates_to_directed_graph,
    create_tsp_mtz_model,
    create_tsp_dfj_model,
)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 solve.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    coordinates = parse_tsp_file(filename)
    graph = coordinates_to_directed_graph(coordinates)
    # model = create_tsp_mtz_model(graph)
    model, cut = create_tsp_dfj_model(graph)
    model.optimize(cut)

    # Draw the graph
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node, pos=graph.nodes[node]["pos"])
    vars = model.getVars()
    x = dict()
    for var in vars:
        if var.varName.startswith("x"):
            x[(int(var.varName.split("_")[1]), int(var.varName.split("_")[2]))] = max(
                0, float(var.x)
            )
    for edge in x:
        if x[edge] > 0.0001:
            G.add_edge(edge[0], edge[1])

    nx.draw(G, nx.get_node_attributes(G, "pos"), with_labels=True)
    nx.draw_networkx_edge_labels(
        G,
        nx.get_node_attributes(G, "pos"),
        edge_labels=nx.get_edge_attributes(G, "weight"),
    )
    print(f"Optimal tour: {model.objVal}")

    # Show the plot
    import matplotlib.pyplot as plt
    from matplotlib import pyplot

    pyplot.gca().invert_yaxis()
    pyplot.gca().invert_xaxis()
    plt.show()
