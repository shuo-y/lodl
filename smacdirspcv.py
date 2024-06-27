import random
import argparse
import time
import numpy as np
import torch
import datetime as dt
import xgboost as xgb

# From SMAC examples

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss, search_weights_loss
from ShortestPath import ShortestPath
from smacdirected import DirectedLoss, QuadSearch, DirectedLossCrossValidation, test_config, test_dir_weight
from utils import perfrandomdq

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
    parser.add_argument("--search-method", type=str, default="mse++", choices=["mse++", "quad"])
    parser.add_argument("--search_estimators", type=int, default=100)
    parser.add_argument("--output", type=str, default="two_quad_example")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=400)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--solver", type=str, choices=["scip", "gurobi", "glpk"], default="scip", help="optimization solver to use")
    parser.add_argument("--numfeatures", type=int, default=5)
    parser.add_argument("--spgrid", type=str, default="(5, 5)")
    parser.add_argument("--param-low", type=float, default=0.001)
    parser.add_argument("--param-upp", type=float, default=2.5)
    parser.add_argument("--param-def", type=float, default=0.05)
    parser.add_argument("--test-history", action="store_true", help="Check test dl of the history")
    parser.add_argument("--cross-valid", action="store_true", help="Use cross validation during search")
    parser.add_argument("--cv-fold", type=int, default=5)

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
    # Y shape *, 40  X shape *, 5
    X, Y = prob.generate_dataset(N=args.num_train + args.num_val + args.num_test, deg=6, noise_width=0.5)


    total_num = args.num_train + args.num_val + args.num_test
    indices = list(range(total_num))
    np.random.shuffle(indices)

    xtrain = X[indices[:params["num_train"]]]
    ytrain = Y[indices[:params["num_train"]]]


    xval = X[indices[params["num_train"] : (params["num_train"] + params["num_val"])]]
    yval = Y[indices[params["num_train"] : (params["num_train"] + params["num_val"])]]


    xtrainvalall = X[indices[:(params["num_train"] + params["num_val"])]]
    ytrainvalall = Y[indices[:(params["num_train"] + params["num_val"])]]


    xtest = X[indices[(args.num_train + args.num_val):]]
    ytest = Y[indices[(args.num_train + args.num_val):]]


    # Check a baseline first
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrainvalall, ytrainvalall)

    ytrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(ytrainpred, ytrain).flatten()

    yvalpred = reg.predict(xval)
    valdl2st = prob.dec_loss(yvalpred, yval).flatten()

    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest).flatten()

    traindltrue = prob.dec_loss(ytrain, ytrain).flatten()
    valdltrue = prob.dec_loss(yval, yval).flatten()
    testdltrue = prob.dec_loss(ytest, ytest).flatten()


    print(f"2st(trained on both train and val) train test val obj, {traindl2st.mean()}, {compute_stderror(traindl2st)}, "
          f"{testdl2st.mean()}, {compute_stderror(testdl2st)}, "
          f"{valdl2st.mean()}, {compute_stderror(valdl2st)}, ")

    _, testdl, teststderr = test_dir_weight(params, prob, xtrain, ytrain, xtest, ytest, None)
    print(f"Def test def directed weight, {testdl}, {teststderr}")

    # The shape of decision is the same as label Y
    traindlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytrain).float(), Y_aux=None, trials=10).numpy().flatten()
    testdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytest).float(), Y_aux=None, trials=10).numpy().flatten()
    valdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(yval).float(), Y_aux=None, trials=10).numpy().flatten()

    search_map = {"mse++": DirectedLoss, "quad": QuadSearch}
    search_map_cv = {"mse++": DirectedLossCrossValidation}


    search_model = search_map_cv[args.search_method]
    model = search_model(prob, params, xtrainvalall, ytrainvalall, args.param_low, args.param_upp, args.param_def, nfold=params["cv_fold"])

    scenario = Scenario(model.configspace, n_trials=args.n_trials)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)
    smac = HPOFacade(scenario, model.train, intensifier=intensifier, overwrite=True)


    records = []
    start_time = time.time()
    for cnt in range(params["n_trials"]):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost)
        records.append((value.cost, info.config))

        smac.tell(info, value)

        if args.test_history:
            _, trainvaldl, trainvaldlstderr = test_config(params, prob, model.get_xgb_params(),  model.get_loss_fn(info.config), xtrainvalall, ytrainvalall, xtest, ytest, auxtest)
            _, testdl, teststderr = test_config(params, prob, model.get_xgb_params(), model.get_loss_fn(info.config), xtrain, ytrain, xtest, ytest, auxtest)
            print(f"Vec {model.get_vec(info.config)}")
            print(f"history vol test teststderr, {cost}, {trainvaldl}, {trainvaldlstderr}, {testdl}, {teststderr}")

    print(f"Search takes {time.time() - start_time} seconds")

    records.sort()
    incumbent = records[0][1]
    params_vec = model.get_vec(incumbent)
    print(f"Seaerch Choose {params_vec}")
    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrainvalall, ytrainvalall)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain).flatten()

    smacyvalpred = booster.inplace_predict(xval)
    valsmac = prob.dec_loss(smacyvalpred, yval).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest).flatten()


    sanity_check(traindl2st - traindltrue, "train2st")
    sanity_check(valdl2st - valdltrue, "val2stall")
    sanity_check(testdl2st - testdltrue, "test2st")
    sanity_check(trainsmac - traindltrue, "trainsmac")
    sanity_check(valsmac - valdltrue, "valsmacall")
    sanity_check(testsmac - testdltrue, "testsmac")
    sanity_check(traindlrand - traindltrue, "trainrand")
    sanity_check(valdlrand - valdltrue, "valrand")
    sanity_check(testdlrand - testdltrue, "testrand")



    print("DQ, 2stagetrainobj, 2stagetestobj, 2statevalobj, "
          "smactrainobj, smactestobj, smacvalobj, "
          "randtrainobj, randtestobj, randvalobj, "
          "truetrainobj, truetestobj, truevalobj, ")
    print(f"DQ, {-1 * traindl2st.mean()}, {-1 * testdl2st.mean()}, {-1 * valdl2st.mean()}, "
          f"{-1 * trainsmac.mean()}, {-1 * testsmac.mean()}, {-1 * valsmac.mean()}, "
          f"{-1 * traindlrand.mean()}, {-1 * testdlrand.mean()}, {-1 * valdlrand.mean()}, "
          f"{-1 * traindltrue.mean()}, {-1 * testdltrue.mean()}, {-1 * valdltrue.mean()}, ")

    print("NorDQ, 2stagetest, smactest, 2stageval, smacval, 2stagetrain, smactrain, ")
    print(f"NorDQ, {((-testdl2st + testdlrand)/(-testdltrue + testdlrand)).mean()}, {((-testsmac + testdlrand)/(-testdltrue + testdlrand)).mean()}, "
          f"{((-valdl2st + valdlrand)/(-valdltrue + valdlrand)).mean()}, {((-valsmac + valdlrand)/(-valdltrue + valdlrand)).mean()}, "
          f"{((-traindl2st + traindlrand)/(-traindltrue + traindlrand)).mean()}, {((-trainsmac + traindlrand)/(-traindltrue + traindlrand)).mean()}, ")

    import sys
    print(sys.argv[0])


