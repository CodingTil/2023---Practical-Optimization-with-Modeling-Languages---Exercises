#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:37:35 2020

@author: luebbecke
"""

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from gurobipy import *


# some output helpful for debugging
verbose = False

# and the corresponding print method
def vprint(*str):
    if verbose == True:
        print(str)


###############################################################################

# visualization of final solution
def visu_patterns(patterns, lambdas, L, l, n):
    # plot items of each order sequentially, using a bar plot for each

    # colormap
    inferno = plt.cm.get_cmap("inferno", n)

    # top and bottom y-coordinates of "current" order i
    top = [0] * len(patterns)
    bottom = [0] * len(patterns)

    # need to traverse the patterns "horizontally" for bar plots
    for i in range(n):
        # height of "current" order
        height = [0] * len(patterns)

        for p in range(len(patterns)):
            if i in patterns[p]:
                height[p] = l[i] * patterns[p][i]
                top[p] = top[p] + height[p]
        # plot current order i
        plt.bar(list(range(len(patterns))), height, bottom=bottom, color=inferno(i / n))
        # old top becomes new bottom, for "next" order i+1
        bottom = top[:]

    # assumes that solve method is called from data file :(
    plt.savefig("cs-%s.png" % (sys.argv[0].split(".")[0]), dpi=300, transparent=True)

    plt.show()


###############################################################################


def solve(m, L, d, l):

    # number of orders
    n = len(d)

    # aggregate data if we have several orders of identical length
    agg = {}  # dictionary with length : demand pairs
    for i in range(n):
        if l[i] not in agg:
            agg[l[i]] = d[i]
            continue
        agg[l[i]] += d[i]

    # now update the input data
    l = list(agg.keys())
    d = list(agg.values())
    n = len(d)

    # create initial set of cutting patterns (we maintain this list only for housekeeping and visualization)
    # there is one pattern/roll per order; the order is cut from that roll as often as it fits
    patterns = []
    for i in range(n):
        # a pattern is (sparsely) stored as a dictionary with keys for orders; values for cutting frequencies
        patterns.append({i: int(L / l[i])})
        # patterns.append({i : 1}) # alternative

    vprint("initial patterns:", patterns)

    # build the initial restricted master problem (RMP)
    rmp = Model("cutting stock RMP")

    # one (continuous) variable per pattern
    lambdas = {}
    for p in range(len(patterns)):
        lambdas[p] = rmp.addVar(obj=1.0, name=f"lambda_{p}")

    # satisfy demands of orders
    demand = {}
    for i in range(n):
        demand[i] = rmp.addConstr(
            quicksum(
                patterns[p][i] * lambdas[p]
                for p in range(len(patterns))
                if i in patterns[p]
            )
            >= d[i],
            name=f"demand_{i}",
        )

    # gurobi should be silent
    rmp.params.outputFlag = 0
    rmp.write("RMP.lp")

    # build pricing problem
    pp = Model("cutting stock pricing problem")

    # one integer variable per order (how often should it be cut in a pattern?)
    x = {}
    for i in range(n):
        x[i] = pp.addVar(vtype=GRB.INTEGER, name=f"x_{i}")

    # knapsack constraint for the roll length
    pp.addConstr(quicksum(l[i] * x[i] for i in range(n)) <= L)

    # gurobi should be silent
    pp.params.outputFlag = 0
    pp.write("PP.lp")

    # column generation loop
    optimal = False
    iteration = 0

    while not optimal:
        iteration += 1
        vprint(f"\niteration {iteration}")

        # re-optimize the RMP
        rmp.optimize()
        vprint(f"curremt RMP objective value is {rmp.objval:.6f}")

        # obtain dual optimal solution
        pi = {}
        for i in range(n):
            pi[i] = demand[i].getAttr(GRB.Attr.Pi)  # demand[i].pi works as well

        # debugging: output dual variable values
        for i in range(n):
            vprint(f"pi[{i}] is {pi[i]:.4f}")

        # modify pricing problem's objective
        pp.setObjective(1 + quicksum(-pi[i] * x[i] for i in range(n)))

        # solve pricing problem
        pp.optimize()

        vprint(f"pricing: smallest reduced cost is {pp.objVal:.6f}")
        if pp.objVal < -0.00001:

            # create a new pattern
            new_pattern = {}
            for i in range(n):
                if x[i].x > 0.1:  # sparsely store the pattern, only positive entries
                    new_pattern[i] = x[i].x
            patterns.append(new_pattern)

            # create a new variable for the new pattern, and add it to the RMP
            # with column coefficients corresponding to the new pattern
            #
            p = len(patterns) - 1
            vprint(f"created pattern {p}: {patterns[p]}")
            lambdas[p] = rmp.addVar(
                obj=1.0,
                name=f"lambda_{p}",
                column=Column(
                    list(int(x[i].x) for i in range(n)),
                    list(demand[i] for i in range(n)),
                ),
            )

            # visu_patterns(patterns, lambdas, L, l, n)

        else:  # no negative reduced cost column found
            vprint("pricing: no negative reduced cost column found")
            optimal = True

        # pp.write(f"PP-{iteration:04d}.lp")
        # rmp.write(f"RMP-{iteration:04d}.lp")

    # output optimal solution objective function value of LP relaxation
    vprint(f"\noptimal objective at the root node: {rmp.objval}")
    for p in range(len(patterns)):
        if lambdas[p].x > 0.001:
            vprint(f"lambda[{p}] = {lambdas[p].x:.4f}")

    # change all RMP variables to integers
    for j in rmp.getVars():
        j.vtype = GRB.INTEGER

    # and start a branch-and-bound search
    # this is a heuristic only, since we do not generate patterns down in the tree
    rmp.optimize()

    # output final (heuristic) integer solution and objective function
    print(f"\nobjective (potentially suboptimal) after B&B: {rmp.objval}")
    for p in range(len(patterns)):
        if lambdas[p].x > 0.001:
            print(f"lambda[{p}] = {lambdas[p].x:.4f}")

    # solution checker (paranoia!)
    vprint("\n*** verfiying solution")
    vprint(f"data: L={L}, l: {l}, d: {d}")

    vprint(f"number of rolls used: {sum(lambdas[p].x for p in range(len(patterns)))}")
    for i in range(n):
        vprint(
            f"order {i} is cut {sum(patterns[p][i]*lambdas[p].x for p in range(len(patterns)) if i in patterns[p])} times; demand is {d[i]}"
        )
        for p in range(len(patterns)):
            if lambdas[p].x > 0.001 and i in patterns[p]:
                vprint(
                    f"- pattern {p}={patterns[p]} used {lambdas[p].x} times, contributes {patterns[p][i] * lambdas[p].x} items"
                )

    print(f"\nPatterns generated: {len(patterns)} (including {n} initial patterns)")
    print(patterns)

    visu_patterns(patterns, lambdas, L, l, n)
    return


#############*****************************************************************************************************************************


def solve_old(m, rollSize, demands, sizes):
    # check consistency
    assert len(sizes) == len(demands)
    numItems = len(sizes)

    # restricted master problem for cutting stock
    model = Model("RMP", Env(""))

    model.params.outputFlag = 0

    # variables:
    #
    # x[p]: "lambdas", how often is pattern p used?
    # pattern[p][i] = "a_ip", stores how often item i is cut in pattern p

    # initalize RMP with trivial cutting patterns,
    # one for each demand
    pattern = []
    x = []
    for p in range(numItems):
        pattern.append([0] * numItems)
        pattern[p][p] = int(rollSize / sizes[p])
        x.append(
            model.addVar(lb=0.0, ub=GRB.INFINITY, name="pattern_" + str(p), obj=1.0)
        )

    # minimize number of used rolls
    model.modelSense = GRB.MINIMIZE

    # constraints:
    #
    # "master constraints", fulfill all demands
    cons = []
    for i in range(numItems):
        cons.append(
            model.addConstr(
                quicksum(pattern[p][i] * x[p] for p in range(len(pattern)))
                >= demands[i]
            )
        )

    solveIP = False  # True <=> RMP is optimally solved
    iter = 0
    while True:
        iter += 1
        model.update()

        # debugging
        # model.write("RMP.lp")

        model.optimize()

        # output (intermediate) solution of the RMP
        if model.status == GRB.status.OPTIMAL:
            for p in range(len(pattern)):
                print("pattern %d is used %g times:" % (p, x[p].x))
                for i in range(numItems):
                    print(
                        "  item %d (of size %d): cut %d times (demand = %d)"
                        % (i, sizes[i], pattern[p][i], demands[i])
                    )
        else:
            print("RMP is infeasible.")

        if solveIP:
            break

        # output dual variables
        for i in range(numItems):
            print(
                "dual variable value for demand %d has a value of %g." % (i, cons[i].pi)
            )

        print("\n\ngenerating pricing problem (PP).")

        # a second model is necessary for the pricing problem
        # this is a knapsack problem in this case (for cutting stock)
        pricingModel = Model("PP", Env(""))

        # saves us from verbose Gurobi output when solving the PP
        pricingModel.params.outputFlag = 0

        # the PP objective function is 1 + max ..., i.e. it contains a constant (1)
        pricingModel.setAttr("objCon", 1.0)

        # one variable ("x_i") per demand, its objective coefficient is -pi_i.
        px = {}
        for i in range(numItems):
            px[i] = pricingModel.addVar(
                lb=0.0,
                ub=GRB.INFINITY,
                name="px_" + str(i),
                obj=-1.0 * cons[i].pi,
                vtype=GRB.INTEGER,
            )

        pricingModel.modelSense = GRB.MINIMIZE
        pricingModel.update()
        pricingModel.write("pp-%d.lp" % (iter))

        # a single knapsack constraint
        pricingModel.addConstr(
            quicksum(sizes[i] * px[i] for i in range(numItems)) <= rollSize
        )

        # solve PP (integrally), this yields an attractive pattern or proves that none exists
        pricingModel.optimize()
        if pricingModel.status != GRB.status.OPTIMAL:
            raise Exception("could not solve pricing problem!")

        # check PP's objective function value for negativity
        if pricingModel.objval < -0.001:  # numerically safer
            print("\n\nfound new cutting pattern!\n")
            pat = [int(px[i].x + 0.5) for i in range(numItems)]

            pattern.append(pat)
            column = Column(pat, cons)
            x.append(
                model.addVar(
                    lb=0.0,
                    ub=GRB.INFINITY,
                    name="pattern_" + str(len(x)),
                    obj=1.0,
                    column=Column(pat, cons),
                )
            )
        else:
            print(
                "\n\nno new cutting patterns found; solving the master problem integrally."
            )

            # change type of all variables from continuous to integer
            for i in range(len(x)):
                x[i].vtype = GRB.INTEGER
            solveIP = True
        print(
            "objective function value (number of rolls): %f"
            % (model.getObjective().getValue())
        )
    return model
