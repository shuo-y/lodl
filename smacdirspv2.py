import random
import argparse
import numpy as np
import torch
import xgboost as xgb
import time

# From SMAC examples

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss, search_weights_loss
from ShortestPath import ShortestPath
from smacdirected import DirectedLoss, QuadSearch, DirectedLossMag, test_config

def compute_stderror(vec: np.ndarray) -> float:
    popstd = vec.std()
    n = len(vec)
    return (popstd * np.sqrt(n / (n - 1.0))) / np.sqrt(n)

def sanity_check(vec: np.ndarray, msg: str) -> None:
    if (vec < 0).any():
        print(f"{msg}: check some negative value {vec}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--search-method", type=str, default="mse++", choices=["mse++", "msemag++", "quad"])
    parser.add_argument("--search_estimators", type=int, default=100)
    parser.add_argument("--output", type=str, default="two_quad_example")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-heldval", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=400)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--solver", type=str, choices=["scip", "gurobi", "glpk"], default="scip", help="optimization solver to use")
    parser.add_argument("--numfeatures", type=int, default=4)
    parser.add_argument("--spgrid", type=str, default="(5, 5)")
    parser.add_argument("--param-low", type=float, default=0.1)
    parser.add_argument("--param-upp", type=float, default=10.0)
    parser.add_argument("--param-def", type=float, default=1.0)
    parser.add_argument("--test-history", action="store_true", help="Check test dl of the history")
    parser.add_argument("--topk", type=int, default=50, help="Another held out set")

    args = parser.parse_args()
    params = vars(args)

    # Load problem
    print(f"Hyperparameters: {args}\n")


    random.seed(args.seed)
    np.random.seed(args.seed * 1091)
    torch.manual_seed(args.seed * 2621)

    params = vars(args)


    prob = ShortestPath(num_feats=args.numfeatures,
                               grid=eval(args.spgrid),
                               solver=args.solver)
    # Generate data based on https://github.com/facebookresearch/LANCER/blob/main/DFL/scripts/run_lancer_dfl.py
    total_num = args.num_train + args.num_val + args.num_heldval + args.num_test
    X, Y = prob.generate_dataset(N=total_num, deg=6, noise_width=0.5)

    indices = list(range(total_num))
    assert args.topk < args.n_trials
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]]


    xval = X[indices[args.num_train:(args.num_train + args.num_val)]] # How does it work for [:50] ??
    yval = Y[indices[args.num_train:(args.num_train + args.num_val)]]

    xheld = X[indices[(args.num_train + args.num_val):(args.num_train + args.num_val + args.num_heldval)]]
    yheld = Y[indices[(args.num_train + args.num_val):(args.num_train + args.num_val + args.num_heldval)]]

    xtest = X[indices[(args.num_train + args.num_val + args.num_heldval):]]
    ytest = Y[indices[(args.num_train + args.num_val + args.num_heldval):]]



    # Check a baseline first
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)


    ytrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(ytrainpred, ytrain).flatten()

    yvalpred = reg.predict(xval)
    valdl2st = prob.dec_loss(yvalpred, yval).flatten()

    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest).flatten()

    helddltrue = prob.dec_loss(yheld, yheld).flatten()

    traindltrue = prob.dec_loss(ytrain, ytrain).flatten()
    valdltrue = prob.dec_loss(yval, yval).flatten()
    testdltrue = prob.dec_loss(ytest, ytest).flatten()

    # The shape of decision is the same as label Y
    traindlrand = -1.0 * prob.get_objective(torch.tensor(ytrain).float(), torch.rand(ytrain.shape)).numpy().flatten()
    valdlrand = -1.0 * prob.get_objective(torch.tensor(yval).float(), torch.rand(yval.shape)).numpy().flatten()
    testdlrand = -1.0 * prob.get_objective(torch.tensor(ytest).float(), torch.rand(ytest.shape)).numpy().flatten()

    search_map = {"mse++": DirectedLoss, "quad": QuadSearch, "msemag++": DirectedLossMag}
    search_model = search_map[args.search_method]

    model = search_model(prob, params, xtrain, ytrain, xval, yval, valdltrue, None, args.param_low, args.param_upp, args.param_def)
    scenario = Scenario(model.configspace, n_trials=args.n_trials)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)  # We basically use one seed per config only

    smac = HPOFacade(scenario, model.train, intensifier=intensifier, overwrite=True)

    start_time = time.time()
    # We can ask SMAC which trials should be evaluated next
    history_val = []
    records = []
    for cnt in range(args.n_trials):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost)
        records.append((value.cost, info.config))

        smac.tell(info, value)

        if args.test_history:
            testdl, testvar = test_config(params, prob, model, xtrain, ytrain, xtest, ytest, testdltrue, info.config)
            helddl, heldvar = test_config(params, prob, model, xtrain, ytrain, xheld, yheld, helddltrue, info.config)
            history_val.append((cost, testdl, testvar, helddl, heldvar))

    print(f"Search takes {time.time() - start_time} seconds")

    records.sort()

    heldvalues = []
    for i in range(args.topk):
        conf = records[i][1]
        testdl, _ = test_config(params, prob, model, xtrain, ytrain, xheld, yheld, helddltrue, conf)
        heldvalues.append(testdl)

    pickind = np.argmin(heldvalues)
    incumbent =records[pickind][1]
    params_vec = model.get_vec(incumbent)
    print(f"Final choose {params_vec}")

    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain).flatten()

    smacyvalpred = booster.inplace_predict(xval)
    valsmac = prob.dec_loss(smacyvalpred, yval).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest).flatten()


    sanity_check(traindl2st - traindltrue, "train2st")
    sanity_check(valdl2st - valdltrue, "val2st")
    sanity_check(testdl2st - testdltrue, "test2st")
    sanity_check(trainsmac - traindltrue, "trainsmac")
    sanity_check(valsmac - valdltrue, "valsmac")
    sanity_check(testsmac - testdltrue, "testsmac")
    sanity_check(traindlrand - traindltrue, "trainrand")
    sanity_check(valdlrand - valdltrue, "valrand")
    sanity_check(testdlrand - testdltrue, "testrand")


    res_str= [(f"res,2stageTrainDL,2stageTrainDLstderr,2stageValDL,2stageValDLstderr,2stageTestDL,2stageTestDLstderr,"
               f"smacTrainDL,smacTrainDLsstderr,smacValDL,smacValDLstderr,smacTestDL,smacTestDLstderr,"
               f"randTrainDL,randTrainDLsstderr,randValDL,randValDLstderr,randTestDL,randTestDLstderr")]
    res_str.append((f"res, {(traindl2st - traindltrue).mean()}, {compute_stderror(traindl2st - traindltrue)}, "
                    f"{(valdl2st - valdltrue).mean()}, {compute_stderror(valdl2st - valdltrue)}, "
                    f"{(testdl2st - testdltrue).mean()}, {compute_stderror(testdl2st - testdltrue)}, "
                    f"{(trainsmac - traindltrue).mean()}, {compute_stderror(trainsmac - traindltrue)}, "
                    f"{(valsmac - valdltrue).mean()}, {compute_stderror(valsmac - valdltrue)}, "
                    f"{(testsmac - testdltrue).mean()}, {compute_stderror(testsmac - testdltrue)}, "
                    f"{(traindlrand - traindltrue).mean()}, {compute_stderror(traindlrand - traindltrue)}, "
                    f"{(valdlrand - valdltrue).mean()}, {compute_stderror(valdlrand - valdltrue)}, "
                    f"{(testdlrand - testdltrue).mean()}, {compute_stderror(testdlrand - testdltrue)}"))


    for row in res_str:
        print(row)

        #TODO how Lower L map to y0y1


    if args.test_history:
        for valdl, testdl, testvar, helddl, heldvar in history_val:
            print(f"{valdl},{testdl},{testvar},{helddl},{heldvar}")



