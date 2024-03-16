#!/usr/bin/env python3
from gurobipy import *
import json
import networkx as nx
from networkx.readwrite import json_graph


def build_graph(g_street, jobs):
    """Constructs the time-expanded graph

    Args:
        g_street nx.DiGraph: The Street Graph
        jobs Dictionary: Jobs and their Information

    Returns:
        nx.DiGraph: what do you want to return?
    """

    # New directed graph
    g_time_expanded = nx.DiGraph()

    deadline = determine_max_time(jobs)

    # Create a street map for each time step
    for t in range(deadline + 1):
        # copy the nodes with data "pos"
        for node in g_street.nodes:
            pos = g_street.nodes[node]["pos"]
            # based on time, shift the position of the nodes up
            pos = (pos[0] + t * 0.5, pos[1] + t * 1.2)
            g_time_expanded.add_node((node, t), pos=pos)

    # Adding time arcs based on "weight"
    for edge in g_street.edges:
        duration = g_street.edges[edge]["weight"]
        for t in range(deadline - duration + 1):
            g_time_expanded.add_edge(
                (edge[0], t), (edge[1], t + duration), weight=duration
            )
            g_time_expanded.add_edge(
                (edge[1], t), (edge[0], t + duration), weight=duration
            )

    # Add waiting time arcs
    for t in range(deadline):
        for node in g_street.nodes:
            g_time_expanded.add_edge((node, t), (node, t + 1), weight=1)

    # for each job add a sink node from each node in the graph
    for job in jobs.keys():
        job_deadline = jobs[job]["j_d"]
        job_release_time = jobs[job]["j_r"]
        job_target_node = jobs[job]["j_t"]
        g_time_expanded.add_node(job, pos=(-1, int(job_deadline)))
        for t in range(job_release_time, job_deadline + 1):
            g_time_expanded.add_edge((job_target_node, t), job, weight=0)

    return g_time_expanded


# You do not need to change anything in this function
def read_instance(full_instance_path):
    """Reads JSON file

    Args:
        full_instance_path (string): Path to the instance

    Returns:
        Dictionary: Jobs
        nx.DiGraph: Graph of the street network
    """
    with open(full_instance_path) as f:
        data = json.load(f)
    return (data["jobs"], json_graph.node_link_graph(data["graph"]))


# Lots of work to do in this function!
def solve(full_instance_path):
    """Solving function, takes an instance file, constructs the time-expanded graph, builds and solves a gurobi model

    Args:
        full_instance_path (string): Path to the instance file to read in

    Returns:
        model: Gurobi model after solving
        G: time-expanded graph
    """

    # Read in the instance data
    jobs, g_street = read_instance(full_instance_path)

    # Construct graph
    g_time_expanded = build_graph(g_street, jobs)

    # === Gurobi model ===
    model = Model("AGV")

    # --- Variables ---

    # Commodity arc variables
    x = {}  # --- TODO ---
    for edge in g_time_expanded.edges:
        for job in jobs.keys():
            # x_((0,1),(1,3))_j
            # if not isinstance(edge[0], tuple) or not isinstance(edge[1], tuple):
            # if type(edge[0]) is not tuple or type(edge[1]) is not tuple:
            if not is_tuple(edge[0]) or not is_tuple(edge[1]):
                # is link to sink
                name = "y_{}_{}".format(edge, job)
            else:
                # is in A'
                name = "x_{}_{}".format(edge, job)
            # remove whitespaces
            name = name.replace(" ", "")
            # print(name)
            x[edge, job] = model.addVar(vtype=GRB.BINARY, name=name)

    # Potentially additional variables? --- TODO ---

    model.update()

    # --- Constraints
    # A street can be used only once at a time
    checked_edges = set()
    for edge in g_time_expanded.edges:
        start, end = edge
        # if not isinstance(start, tuple) or not isinstance(end, tuple):
        # if type(start) is not tuple or type(end) is not tuple:
        if not is_tuple(start) or not is_tuple(end):
            # link to sink
            continue
        start_node, start_time = start
        end_node, end_time = end
        duration = g_time_expanded.edges[edge]["weight"]
        assert duration == end_time - start_time
        reverse_street = ((end_node, start_time), (start_node, end_time))
        if edge in checked_edges or reverse_street in checked_edges:
            # already added constraint for this
            continue
        checked_edges.add(edge)
        checked_edges.add(reverse_street)
        # get all edges during that duration
        edges = set()
        for t in range(start_time, end_time):
            possible_edge = ((start_node, t), (end_node, t + duration))
            if possible_edge in g_time_expanded.edges:
                edges.add(possible_edge)
            possible_reverse_edge = ((end_node, t), (start_node, t + duration))
            if possible_reverse_edge in g_time_expanded.edges:
                edges.add(possible_reverse_edge)
        # add constraint
        model.addConstr(
            quicksum(x[edge, job] for edge in edges for job in jobs.keys()) <= 1
        )

    # Multiflow constraints
    for job in jobs.keys():
        source = jobs[job]["j_s"]
        target = jobs[job]["j_t"]
        release_time = jobs[job]["j_r"]
        deadline = jobs[job]["j_d"]

        time_graph_source_node = (source, release_time)
        time_graph_sink_node = job

        for node in g_time_expanded.nodes:
            # incoming arcs
            incoming_arcs = set(g_time_expanded.in_edges(node))
            # outgoing arcs
            outgoing_arcs = set(g_time_expanded.out_edges(node))

            # if node is source node
            if node == time_graph_source_node:
                model.addConstr(
                    quicksum(x[out_arc, job] for out_arc in outgoing_arcs) == 1
                )
            # if node is sink node
            elif node == time_graph_sink_node:
                model.addConstr(
                    quicksum(x[in_arc, job] for in_arc in incoming_arcs) == 1
                )
            # if node is not source or sink node
            else:
                if len(outgoing_arcs) > 0 and len(incoming_arcs) > 0:
                    model.addConstr(
                        quicksum(x[out_arc, job] for out_arc in outgoing_arcs)
                        == quicksum(x[in_arc, job] for in_arc in incoming_arcs)
                    )
                elif len(outgoing_arcs) == 0 and len(incoming_arcs) > 0:
                    model.addConstr(
                        quicksum(x[in_arc, job] for in_arc in incoming_arcs) == 0
                    )
                elif len(outgoing_arcs) > 0 and len(incoming_arcs) == 0:
                    model.addConstr(
                        quicksum(x[out_arc, job] for out_arc in outgoing_arcs) == 0
                    )

    model.update()

    # --- Objective
    # Minimize required time steps
    model.setObjective(
        quicksum(
            g_time_expanded.edges[edge]["weight"] * x[edge, job]
            for edge in g_time_expanded.edges
            for job in jobs.keys()
        ),
        GRB.MINIMIZE,
    )

    # Solve the model
    model.update()
    model.optimize()
    # model.write("model.lp")
    # If your model is infeasible (but you expect it to not be), comment out the lines below to compute and write out a infeasible subsystem (Might take very long)
    # model.computeIIS()
    # model.write("model.ilp")

    return model, g_time_expanded


def determine_max_time(jobs):
    """Determines the maximum time (deadline) of all jobs

    Args:
        jobs (Dictionary): Jobs and their information

    Returns:
        int: Maximum time of all jobs
    """
    max_time = 0
    for _, obj in jobs.items():
        if obj["j_d"] > max_time:
            max_time = obj["j_d"]
    assert max_time > 0
    return max_time


def is_tuple(x, values_to_expect=2):
    """Checks if x is a tuple, in a very stupid fashion, because tutOR wont allow isinstance(x, tuple) or tuple(x) is tuple or tuple(x) == tuple!

    Args:
        x (any): Object to check
        values_to_expect (int, optional): Number of values to expect in the tuple. Defaults to 2.

    Returns:
        bool: True if x is a tuple, False otherwise
    """
    try:
        return tuple(x) == x and len(x) == values_to_expect
    except:
        return False
