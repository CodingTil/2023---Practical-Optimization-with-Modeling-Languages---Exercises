from vc_bc import solve
from gurobipy import *


def run_test(instance_path):
    print("Start cut testing.")

    model, cuts = solve(instance_path=instance_path, color_largest_clique=True)

    # reset Gurobi model and surpress log
    model.reset()
    model.setParam("LogToConsole", 0)

    correct = True

    # relax the model
    for v in model.getVars():
        v.vtype = GRB.CONTINUOUS

    # go through list of cuts
    for i, it in enumerate(cuts):

        # fix model variables to solution found in callback
        for v, j in it["x_cb_sol"]:
            model.getVarByName(f"x_{v}_{j}").lb = it["x_cb_sol"][v, j]
            model.getVarByName(f"x_{v}_{j}").ub = it["x_cb_sol"][v, j]

        for j in it["y_cb_sol"]:
            model.getVarByName(f"y_{j}").lb = it["y_cb_sol"][j]
            model.getVarByName(f"y_{j}").ub = it["y_cb_sol"][j]

        # optimize the model
        model.optimize()

        # if the model is infeasible, you made a mistake
        if model.Status == GRB.INFEASIBLE:
            print(f"Model is not feasible for LP relax. solution in iteration {i}")
            correct = False

        if correct:
            # check for all cuts if they invalidate the current LP solution
            for c in it["cuts"]:
                cut = model.addConstr(c)
                model.optimize()
                if not model.Status == GRB.INFEASIBLE:
                    print(f"Cut does not cut off LP relax. solution")
                    correct = False
                    break
                else:
                    model.remove(cut)

        if not correct:
            break
    if correct:
        print("Model implemented cut separation correctly")
        return True
    else:
        print("Model did not implement cut separation correctly")
        return False


if __name__ == "__main__":
    run_test("vc_bc_data3.adjlist")
