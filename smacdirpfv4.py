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
from PortfolioOpt import PortfolioOpt
from smacdirected import DirectedLoss, QuadSearch, test_config
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
    parser.add_argument("--num-frac", type=float, default=0.1)
    parser.add_argument("--num-test", type=int, default=400)
    parser.add_argument("--start-time", type=str, default="dt.datetime(2004, 1, 1)")
    parser.add_argument("--end-time", type=str, default="dt.datetime(2017, 1, 1)")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-trials-2nd", type=int, default=20)
    parser.add_argument("--stocks", type=int, default=50)
    parser.add_argument("--stockalpha", type=float, default=0.1)
    parser.add_argument("--param-low", type=float, default=0.0001)
    parser.add_argument("--param-upp", type=float, default=0.01)
    parser.add_argument("--param-def", type=float, default=0.001)
    parser.add_argument("--test-history", action="store_true", help="Check test dl of the history")

    args = parser.parse_args()
    params = vars(args)

    # Load problem
    print(f"Hyperparameters: {args}\n")


    random.seed(args.seed)
    np.random.seed(args.seed * 1091)
    torch.manual_seed(args.seed * 2621)

    params = vars(args)


    prob = PortfolioOpt(num_train_instances = params["num_train"],
                        num_val = params["num_val"],
                        num_test_instances = params["num_test"],
                        num_stocks = params["stocks"],
                        alpha = args.stockalpha,
                        start_time = eval(params["start_time"]),
                        end_time = eval(params["end_time"]))

    X, Y, Aux = prob.get_np_data()
    X = X.reshape(X.shape[0], np.prod(X.shape[1:]))

    total_num = args.num_train + args.num_val + args.num_test
    indices = list(range(total_num))
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]]
    auxtrain = Aux[indices[:args.num_train]]

    xvalall = X[indices[args.num_train:(args.num_train + args.num_val)]]
    yvalall = Y[indices[args.num_train:(args.num_train + args.num_val)]]
    auxvalall = Aux[indices[args.num_train:(args.num_train + args.num_val)]]

    xval = xvalall[:int(params["num_frac"] * params["num_val"])]
    yval = yvalall[:int(params["num_frac"] * params["num_val"])]
    auxval = auxvalall[:int(params["num_frac"] * params["num_val"])]


    xtest = X[indices[(args.num_train + args.num_val):]]
    ytest = Y[indices[(args.num_train + args.num_val):]]
    auxtest = Aux[indices[(args.num_train + args.num_val):]]


    # Check a baseline first
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)


    ytrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(ytrainpred, ytrain, aux_data=auxtrain).flatten()

    yvalpred = reg.predict(xval)
    valdl2st = prob.dec_loss(yvalpred, yval, aux_data=auxval).flatten()

    yvalallpred = reg.predict(xvalall)
    valdl2stall = prob.dec_loss(yvalallpred, yvalall, aux_data=auxvalall).flatten()

    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest, aux_data=auxtest).flatten()

    traindltrue = prob.dec_loss(ytrain, ytrain, aux_data=auxtrain).flatten()
    valdltrue = prob.dec_loss(yval, yval, aux_data=auxval).flatten()
    testdltrue = prob.dec_loss(ytest, ytest, aux_data=auxtest).flatten()
    valdltrueall = prob.dec_loss(yvalall, yvalall, aux_data=auxvalall).flatten()

    print(f"2st train test val valall obj, {traindl2st.mean()}, {compute_stderror(traindl2st)}, "
          f"{testdl2st.mean()}, {compute_stderror(testdl2st)}, "
          f"{valdl2st.mean()}, {compute_stderror(valdl2st)}, "
          f"{valdl2stall.mean()}, {compute_stderror(valdl2stall)}, ")

    # The shape of decision is the same as label Y
    traindlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytrain), Y_aux=torch.tensor(auxtrain), trials=10).numpy().flatten()
    testdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytest), Y_aux=torch.tensor(auxtest), trials=10).numpy().flatten()
    valdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(yval), Y_aux=torch.tensor(auxval), trials=10).numpy().flatten()
    valdlrandall = -1.0 * perfrandomdq(prob, Y=torch.tensor(yvalall), Y_aux=torch.tensor(auxvalall), trials=10).numpy().flatten()

    search_map = {"mse++": DirectedLoss, "quad": QuadSearch}
    search_model = search_map[args.search_method]

    model = search_model(prob, params, xtrain, ytrain, xval, yval, valdltrue, auxval, args.param_low, args.param_upp, args.param_def, reg2st=reg)
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
            testdl, teststderr = test_config(params, prob, model, xtrain, ytrain, xtest, ytest, auxtest, info.config)
            print(f"history vol test teststderr, {cost}, {testdl}, {teststderr}")

    print(f"Search takes {time.time() - start_time} seconds")

    records.sort()
    incumbent = records[0][1]
    params_vec = model.get_vec(incumbent)
    print(f"1st step Choose {params_vec}")

    model = search_model(prob, params, xtrain, ytrain, xvalall, yvalall, valdltrueall, auxvalall, args.param_low, args.param_upp, args.param_def, use_vec=True, initvec=params_vec)
    scenario = Scenario(model.configspace, n_trials=args.n_trials_2nd)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)
    smac = HPOFacade(scenario, model.train, intensifier=intensifier, overwrite=True)

    records2 = []
    start_time = time.time()
    for cnt in range(params["n_trials_2nd"]):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost)
        records2.append((value.cost, info.config))

        smac.tell(info, value)

        if args.test_history:
            testdl, teststderr = test_config(params, prob, model, xtrain, ytrain, xtest, ytest, auxtest, info.config)
            print(f"2nd step history vol test teststderr, {cost}, {testdl}, {teststderr}")

    print(f"2nd step search takes {time.time() - start_time} seconds")

    records2.sort()
    incumbent = records2[0][1]
    params_vec = model.get_vec(incumbent)
    print(f"2nd step Choose {params_vec}")

    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain, aux_data=auxtrain).flatten()

    smacyvalpred = booster.inplace_predict(xval)
    valsmac = prob.dec_loss(smacyvalpred, yval, aux_data=auxval).flatten()

    smacyvalallpred = booster.inplace_predict(xvalall)
    valsmacall = prob.dec_loss(smacyvalallpred, yvalall, aux_data=auxvalall).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest, aux_data=auxtest).flatten()



    sanity_check(traindl2st - traindltrue, "train2st")
    sanity_check(valdl2stall - valdltrueall, "val2stall")
    sanity_check(testdl2st - testdltrue, "test2st")
    sanity_check(trainsmac - traindltrue, "trainsmac")
    sanity_check(valsmacall - valdltrueall, "valsmacall")
    sanity_check(testsmac - testdltrue, "testsmac")
    sanity_check(traindlrand - traindltrue, "trainrand")
    sanity_check(valdlrand - valdltrue, "valrand")
    sanity_check(testdlrand - testdltrue, "testrand")



    print("DQ, 2stagetrainobj, 2stagetestobj, 2statevalobj, 2stagevalallobj, "
          "smactrainobj, smactestobj, smacvalobj, smacvalallobj, "
          "randtrainobj, randtestobj, randvalobj, randvalallobj, "
          "truetrainobj, truetestobj, truevalobj, truevalallobj, ")
    print(f"DQ, {-1 * traindl2st.mean()}, {-1 * testdl2st.mean()}, {-1 * valdl2st.mean()}, {-1 * valdl2stall.mean()}, "
          f"{-1 * trainsmac.mean()}, {-1 * testsmac.mean()}, {-1 * valsmac.mean()}, {-1 * valsmacall.mean()}, "
          f"{-1 * traindlrand.mean()}, {-1 * testdlrand.mean()}, {-1 * valdlrand.mean()}, {-1 * valdlrandall.mean()}, "
          f"{-1 * traindltrue.mean()}, {-1 * testdltrue.mean()}, {-1 * valdltrue.mean()}, {-1 * valdltrueall.mean()}, ")

    print("NorDQ, 2stagetest, smactest, 2stageval, smacval, 2stagetrain, smactrain, 2stagevalall, smacvalall, ")
    print(f"NorDQ, {((-testdl2st + testdlrand)/(-testdltrue + testdlrand)).mean()}, {((-testsmac + testdlrand)/(-testdltrue + testdlrand)).mean()}, "
          f"{((-valdl2st + valdlrand)/(-valdltrue + valdlrand)).mean()}, {((-valsmac + valdlrand)/(-valdltrue + valdlrand)).mean()}, "
          f"{((-traindl2st + traindlrand)/(-traindltrue + traindlrand)).mean()}, {((-trainsmac + traindlrand)/(-traindltrue + traindlrand)).mean()}, "
          f"{((-valdl2stall + valdlrandall)/(-valdltrueall + valdlrandall)).mean()}, {((-valsmacall + valdlrandall)/(-valdltrueall + valdlrandall)).mean()}, ")



