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
    parser.add_argument("--num-heldval", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=400)
    parser.add_argument("--start-time", type=str, default="dt.datetime(2004, 1, 1)")
    parser.add_argument("--end-time", type=str, default="dt.datetime(2017, 1, 1)")
    parser.add_argument("--n-trials", type=int, default=200)
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
                        num_val = params["num_val"] + params["num_heldval"],
                        num_test_instances = params["num_test"],
                        num_stocks = params["stocks"],
                        alpha = args.stockalpha,
                        start_time = eval(params["start_time"]),
                        end_time = eval(params["end_time"]))

    X, Y, Aux = prob.get_np_data()
    X = X.reshape(X.shape[0], np.prod(X.shape[1:]))

    total_num = args.num_train + args.num_val + args.num_heldval + args.num_test
    indices = list(range(total_num))
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]]
    auxtrain = Aux[indices[:args.num_train]]

    xval = X[indices[args.num_train:(args.num_train + args.num_val)]]
    yval = Y[indices[args.num_train:(args.num_train + args.num_val)]]
    auxval = Aux[indices[args.num_train:(args.num_train + args.num_val)]]

    xheld = X[indices[(args.num_train + args.num_val):(args.num_train + args.num_val + args.num_heldval)]]
    yheld = Y[indices[(args.num_train + args.num_val):(args.num_train + args.num_val + args.num_heldval)]]
    auxheld = Aux[indices[(args.num_train + args.num_val):(args.num_train + args.num_val + args.num_heldval)]]


    xtest = X[indices[(args.num_train + args.num_val + args.num_heldval):]]
    ytest = Y[indices[(args.num_train + args.num_val + args.num_heldval):]]
    auxtest = Aux[indices[(args.num_train + args.num_val + args.num_heldval):]]


    # Check a baseline first
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)


    ytrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(ytrainpred, ytrain, aux_data=auxtrain).flatten()

    yvalpred = reg.predict(xval)
    valdl2st = prob.dec_loss(yvalpred, yval, aux_data=auxval).flatten()

    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest, aux_data=auxtest).flatten()

    yheldpred = reg.predict(xheld)
    held2st = prob.dec_loss(yheldpred, yheld, aux_data=auxheld).flatten()

    traindltrue = prob.dec_loss(ytrain, ytrain, aux_data=auxtrain).flatten()
    valdltrue = prob.dec_loss(yval, yval, aux_data=auxval).flatten()
    testdltrue = prob.dec_loss(ytest, ytest, aux_data=auxtest).flatten()
    helddltrue = prob.dec_loss(yheld, yheld, aux_data=auxheld).flatten()

    print(f"2st train val test held diff, {(traindl2st - traindltrue).mean()}, {compute_stderror(traindl2st - traindltrue)}, "
          f"{(valdl2st - valdltrue).mean()}, {compute_stderror(valdl2st - valdltrue)}, "
          f"{(testdl2st - testdltrue).mean()}, {compute_stderror(testdl2st - testdltrue)}, "
          f"{(held2st - helddltrue).mean()}, {compute_stderror(held2st - helddltrue)}, ")

    # The shape of decision is the same as label Y
    traindlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytrain), Y_aux=torch.tensor(auxtrain), trials=10).numpy().flatten()
    testdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytest), Y_aux=torch.tensor(auxtest), trials=10).numpy().flatten()
    helddlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(yheld), Y_aux=torch.tensor(auxheld), trials=10).numpy().flatten()
    valdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(yval), Y_aux=torch.tensor(auxval), trials=10).numpy().flatten()

    search_map = {"mse++": DirectedLoss, "quad": QuadSearch}
    search_model = search_map[args.search_method]

    model = search_model(prob, params, xtrain, ytrain, xval, yval, valdltrue, auxval, args.param_low, args.param_upp, args.param_def, reg2st=reg)
    scenario = Scenario(model.configspace, n_trials=args.n_trials)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)
    smac = HPOFacade(scenario, model.train, intensifier=intensifier, overwrite=True)

    records = []
    start_time = time.time()
    for cnt in range(args.n_trials):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost)
        records.append((value.cost, info.config))

        smac.tell(info, value)

        if args.test_history:
            testdl, testvar = test_config(params, prob, model, xtrain, ytrain, xtest, ytest, auxtest, info.config)
            helddl, heldvar = test_config(params, prob, model, xtrain, ytrain, xheld, yheld, auxheld, info.config)
            print(f"history, {cost}, {testdl}, {testvar}, {helddl}, {heldvar}")

    print(f"Search takes {time.time() - start_time} seconds")

    records.sort()
    incumbent = records[0][1]
    params_vec = model.get_vec(incumbent)
    print(f"Final choose {params_vec}")

    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain, aux_data=auxtrain).flatten()

    smacyvalpred = booster.inplace_predict(xval)
    valsmac = prob.dec_loss(smacyvalpred, yval, aux_data=auxval).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest, aux_data=auxtest).flatten()

    smacyheldpred = booster.inplace_predict(xheld)
    heldsmac = prob.dec_loss(smacyheldpred, yheld, aux_data=auxheld).flatten()


    sanity_check(traindl2st - traindltrue, "train2st")
    sanity_check(valdl2st - valdltrue, "val2st")
    sanity_check(testdl2st - testdltrue, "test2st")
    sanity_check(trainsmac - traindltrue, "trainsmac")
    sanity_check(valsmac - valdltrue, "valsmac")
    sanity_check(testsmac - testdltrue, "testsmac")
    sanity_check(traindlrand - traindltrue, "trainrand")
    sanity_check(valdlrand - valdltrue, "valrand")
    sanity_check(testdlrand - testdltrue, "testrand")


    print("DLdiff,2stageTrainDL,2stageTrainDLstderr,2stageValDL,2stageValDLstderr,2stageTestDL,2stageTestDLstderr,"
          "smacTrainDL,smacTrainDLsstderr,smacValDL,smacValDLstderr,smacTestDL,smacTestDLstderr,"
          "randTrainDL,randTrainDLsstderr,randValDL,randValDLstderr,randTestDL,randTestDLstderr,"
          "held2st,held2ststderr,heldsmac,heldsmacstderr,heldrand,heldrandstderr")
    print(f"DLdiff, {(traindl2st - traindltrue).mean()}, {compute_stderror(traindl2st - traindltrue)}, "
          f"{(valdl2st - valdltrue).mean()}, {compute_stderror(valdl2st - valdltrue)}, "
          f"{(testdl2st - testdltrue).mean()}, {compute_stderror(testdl2st - testdltrue)}, "
          f"{(trainsmac - traindltrue).mean()}, {compute_stderror(trainsmac - traindltrue)}, "
          f"{(valsmac - valdltrue).mean()}, {compute_stderror(valsmac - valdltrue)}, "
          f"{(testsmac - testdltrue).mean()}, {compute_stderror(testsmac - testdltrue)}, "
          f"{(traindlrand - traindltrue).mean()}, {compute_stderror(traindlrand - traindltrue)}, "
          f"{(valdlrand - valdltrue).mean()}, {compute_stderror(valdlrand - valdltrue)}, "
          f"{(testdlrand - testdltrue).mean()}, {compute_stderror(testdlrand - testdltrue)}, "
          f"{(held2st - helddltrue).mean()}, {compute_stderror(held2st - helddltrue)}, "
          f"{(heldsmac - helddltrue).mean()}, {compute_stderror(heldsmac - helddltrue)}, "
          f"{(helddlrand - helddltrue).mean()}, {compute_stderror(helddlrand - helddltrue)}")

    print("DQ, 2stagetrainobj, 2stagetestobj, 2statevalobj, 2stageheldobj, "
          "smactrainobj, smactestobj, smacvalobj, smacheldobj, "
          "randtrainobj, randtestobj, randvalobj, randheldobj, "
          "truetrainobj, truetestobj, truevalobj, trueheldobj, ")
    print(f"DQ, {-1 * traindl2st.mean()}, {-1 * testdl2st.mean()}, {-1 * valdl2st.mean()}, {-1 * held2st.mean()}, "
          f"{-1 * trainsmac.mean()}, {-1 * testsmac.mean()}, {-1 * valsmac.mean()}, {-1 * heldsmac.mean()}, "
          f"{-1 * traindlrand.mean()}, {-1 * testdlrand.mean()}, {-1 * valdlrand.mean()}, {-1 * helddlrand.mean()}, "
          f"{-1 * traindltrue.mean()}, {-1 * testdltrue.mean()}, {-1 * valdltrue.mean()}, {-1 * helddltrue.mean()}, ")

    print("NorDQ, 2stagetest, smactest, 2stageval, smacval, 2stagetrain, smactrain, 2stageheld, smacheld, ")
    print(f"NorDQ, {((-testdl2st + testdlrand)/(-testdltrue + testdlrand)).mean()}, {((-testsmac + testdlrand)/(-testdltrue + testdlrand)).mean()}, "
          f"{((-valdl2st + valdlrand)/(-valdltrue + valdlrand)).mean()}, {((-valsmac + valdlrand)/(-valdltrue + valdlrand)).mean()}, "
          f"{((-traindl2st + traindlrand)/(-traindltrue + traindlrand)).mean()}, {((-trainsmac + traindlrand)/(-traindltrue + traindlrand)).mean()}, "
          f"{((-held2st + helddlrand)/(-helddltrue + helddlrand)).mean()}, {((-heldsmac + helddlrand)/(-helddltrue + helddlrand)).mean()}, ")



