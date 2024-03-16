import networkx as nx
from gurobipy import Model, GRB, quicksum

# TODO: Add other functions here for processing of the graph
EPSILON = 10 ** (-4)
MAXIMUM_CUTS_PER_COLOR = 50


def is_fractional(x) -> bool:
    return EPSILON < x < 1 - EPSILON


def solve(instance_path, color_largest_clique):
    """Builds the model on vertex coloring and runs the B&C algorithm.

    Args:
        instance_path (string): Path to the adjlist to read in
        color_largest_clique (boolean): iff true, the vertices in the largest clique are colored to improve the dual bound
    Returns:
        model (Gurobi.Model): solved gurobipy model instance
    """

    # read graph from instance path
    G = nx.read_adjlist(instance_path)

    # TODO: Get the data you need from the graph G
    N = G.number_of_nodes()

    # set up the model
    model = Model()

    # variable definition
    x = {}
    y = {}
    # TODO: Add your variables here
    for c in range(0, N):
        y[c] = model.addVar(vtype=GRB.BINARY, name="y_" + str(c))
        for v in G.nodes:
            x[v, c] = model.addVar(vtype=GRB.BINARY, name="x_" + str(v) + "_" + str(c))

    model.update()

    # TODO: Add your constraints here
    # each vertex is colored
    for v in G.nodes:
        model.addConstr(quicksum(x[v, c] for c in range(0, N)) == 1)

    USE_SIMPLE_CONTSRAINTS = False
    if USE_SIMPLE_CONTSRAINTS:
        # adjacent vertices have different colors
        for (u, v) in G.edges:
            for c in range(0, N):
                model.addConstr(x[u, c] + x[v, c] <= y[c])
    else:
        # adjacent vertices have different colors (4.)
        for v in G.nodes:
            neighbors = list(G.neighbors(v))

            if len(neighbors) == 0:
                continue

            clique_partition = []
            neighbors_copy = list(G.neighbors(v))
            covered_vertices = []
            while len(neighbors_copy) > 0:
                subgraph = G.subgraph(neighbors_copy)
                maximal_clique = get_maximal_clique(subgraph)
                if maximal_clique is None:
                    break
                clique_partition.append(maximal_clique)
                covered_vertices.extend(maximal_clique)
                neighbors_copy = [v for v in neighbors_copy if v not in maximal_clique]

            # add not covered vertices to clique partition
            for neighbor in neighbors:
                if neighbor not in covered_vertices:
                    clique_partition.append([neighbor])

            assert sorted(neighbors) == sorted(list(G.neighbors(v)))
            assert sorted([x for clique in clique_partition for x in clique]) == sorted(
                neighbors
            )
            for i1 in range(0, len(clique_partition)):
                for i2 in range(i1 + 1, len(clique_partition)):
                    c1 = clique_partition[i1]
                    c2 = clique_partition[i2]
                    assert set(c1).isdisjoint(set(c2))

            # add constraints for each clique in the partition
            for c in range(0, N):
                model.addConstr(
                    quicksum(x[u, c] for u in neighbors)
                    + x[v, c] * len(clique_partition)
                    <= y[c] * len(clique_partition)
                )

    # must use color if it was assigned
    for c in range(0, N):
        for v in G.nodes:
            model.addConstr(x[v, c] <= y[c])

    # breaking symmetry (1)
    for c in range(1, N):
        model.addConstr(y[c - 1] >= y[c])

    # breaking symmetry (2)
    # dont use color if no vertex is assigned to it
    for c in range(0, N):
        model.addConstr(y[c] <= quicksum(x[v, c] for v in G.nodes))

    ########################
    # Color Largest Clique #
    ########################

    if color_largest_clique:
        maximal_clique = get_maximal_clique(G)
        if maximal_clique is not None:
            # add constraints to assign colors c in {0, ..., len(maximal_clique)-1} to the vertices in the maximal clique, such that for c1, c2 assigned to v1,v2 (x[v1, c1], x[v2, c2]) it holds c1 < c2 if v1 < v2
            for c in range(0, len(maximal_clique)):
                model.addConstr(x[maximal_clique[c], c] == 1)
                for c2 in range(0, len(maximal_clique)):
                    if c != c2:
                        model.addConstr(x[maximal_clique[c2], c] == 0)
                model.addConstr(y[c] == 1)

    model.update()

    # TODO: Define the objective function
    # minimize used colors
    model.setObjective(quicksum(y[c] for c in range(0, N)), GRB.MINIMIZE)

    model.update()

    # list to track LP solutions and cuts
    cuts = []

    def cb(model, where):
        """Callback function that implements the separation of clique and block inequalities.

        Args:
            model: model instance in callback
            where: status of the optimizer

        Returns:
        """
        # only enter cut generation process if optimizer is at optimal status
        if (
            where == GRB.Callback.MIPNODE
            and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL
        ):

            # retrieve solution from solver
            x_cb_sol = dict(model.cbGetNodeRel(x))
            y_cb_sol = dict(model.cbGetNodeRel(y))

            sep_dict = {"x_cb_sol": x_cb_sol, "y_cb_sol": y_cb_sol, "cuts": []}

            # TODO: Implement your cut separation process here

            incumbent = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            lower_bound = model.cbGet(GRB.Callback.MIPNODE_OBJBND)

            if incumbent is not None:
                incumbent = int(incumbent)
                max_colors_used = quicksum(y[c] for c in range(0, N)) <= incumbent
                model.cbCut(max_colors_used)
                for c in range(incumbent, N):
                    expr = y[c] == 0
                    model.cbCut(expr)
            else:
                incumbent = N

            if lower_bound is not None:
                if lower_bound > int(lower_bound):
                    lower_bound = int(lower_bound) + 1
                else:
                    lower_bound = int(lower_bound)
                min_colors_used = quicksum(y[c] for c in range(0, N)) >= lower_bound
                model.cbCut(min_colors_used)
                for c in range(0, lower_bound):
                    expr = y[c] == 1
                    model.cbCut(expr)
            else:
                lower_bound = 0

            #############################
            # Seperation of Clique Cuts #
            #############################

            for c0 in range(0, incumbent):
                current_cuts = 0

                # sort vertices by the values of x_v_c0 in the current LP solution in decreasing order
                vertices_sorted = sorted(
                    G.nodes, key=lambda v: x_cb_sol[v, c0], reverse=True
                )
                len_vertices_sorted = len(vertices_sorted)

                # Start with the highest fractional* value in the list and add the vertex to a candidate clique. Then, following the ranking in the list, add neighbors of vv to the candidate clique if they are also neighbors of all other vertices in the candidate clique, i.e., form a larger clique together.
                for index in range(len_vertices_sorted):
                    vertex = vertices_sorted[index]
                    if current_cuts >= MAXIMUM_CUTS_PER_COLOR:
                        break

                    if not is_fractional(x_cb_sol[vertex, c0]):
                        continue

                    candidate_clique = {vertex}
                    for index2 in range(0, len_vertices_sorted):
                        if index == index2:
                            continue
                        vertex2 = vertices_sorted[index2]
                        vertext2_neighbors = set(G.neighbors(vertex2))
                        if candidate_clique.issubset(vertext2_neighbors):
                            candidate_clique.add(vertex2)

                    # check if the clique inequality is violated
                    if sum(x_cb_sol[v, c0] for v in candidate_clique) > 1 + EPSILON:
                        expr = quicksum(x[v, c0] for v in candidate_clique) <= 1
                        sep_dict["cuts"].append(expr)
                        model.cbCut(expr)
                        current_cuts += 1

            ############################
            # Seperation of Block Cuts #
            ############################
            for v in G.nodes:
                for c0 in range(0, incumbent - 1):
                    block_sum = sum(x_cb_sol[(v, c)] for c in range(c0, incumbent))

                    if block_sum > y_cb_sol[c0] + EPSILON:
                        expr = (
                            quicksum(x[(v, c)] for c in range(c0, incumbent)) <= y[c0]
                        )
                        sep_dict["cuts"].append(expr)
                        model.cbCut(expr)

            # TODO: Whenever you find a violated cut, first build it as a temporary constraint, add the temporary constraint to the sep_dict (for our evaluation) and then add it to the model
            # expr = quicksum(x[..] for ..) <= ? ...
            # sep_dict['cuts'].append(expr)
            # model.cbCut(expr)

            # At end of callback append sep_dict
            cuts.append(sep_dict)

    model.setParam("Presolve", 0)
    model.setParam("Cuts", 0)
    model.optimize(cb)

    return model, cuts


def get_maximal_clique(G):
    maximal_cliques = list(nx.find_cliques(G))
    if len(maximal_cliques) == 0:
        return None

    # only consider cliques with maximal size
    maximal_clique_size = max([len(clique) for clique in maximal_cliques])
    maximal_cliques = [
        clique for clique in maximal_cliques if len(clique) == maximal_clique_size
    ]
    assert len(maximal_cliques) > 0

    # select the clique that contains the vertex with the lowest index. in case there is more than one maximum clique that shares the lowest vertex index, use the second vertex index for those cliques (and so on)
    # sort each clique
    maximal_cliques = [sorted(clique) for clique in maximal_cliques]
    index = 0
    while len(maximal_cliques) > 1:
        assert index < maximal_clique_size
        # get the minimal vertex for this index
        minimal_vertex = min([clique[index] for clique in maximal_cliques])
        # filter out all cliques that do not contain this vertex
        maximal_cliques = [
            clique for clique in maximal_cliques if clique[index] == minimal_vertex
        ]
        # increase index
        index += 1
    assert len(maximal_cliques) == 1

    # get the maximal clique
    maximal_clique = maximal_cliques[0]

    # sort the vertices by their index ascending
    maximal_clique = sorted(maximal_clique)

    return maximal_clique


if __name__ == "__main__":
    solved_model, cuts = solve(
        instance_path="vc_bc_data3.adjlist", color_largest_clique=True
    )
