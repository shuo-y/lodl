import random
import argparse
import numpy as np
import torch
import xgboost as xgb

# From SMAC examples

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss
from PortfolioOpt import PortfolioOpt
from smacdirected import DirectedLoss

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
    parser.add_argument("--search_estimators", type=int, default=100)
    parser.add_argument("--output", type=str, default="two_quad_example")
    parser.add_argument("--loss", type=str, default="quad", choices=["mse", "quad"])
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--num-test", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--quad-alpha", type=float, default=0.0)
    parser.add_argument("--stocks", type=int, default=50)
    parser.add_argument("--stockalpha", type=float, default=0.1)

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
                        alpha = args.stockalpha)

    X, Y, Aux = prob.get_data()

    indices = list(range(args.num_train + args.num_val + args.num_test))
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]].squeeze()

    xval = X[indices[args.num_train:(args.num_train + args.num_val)]]
    yval = Y[indices[args.num_train:(args.num_train + args.num_val)]].squeeze()

    xtest = X[indices[(args.num_train + args.num_val):]]
    ytest = Y[indices[(args.num_train + args.num_val):]].squeeze()

    print("check ytrain")
    prob.checky(ytrain)
    print("check yval")
    prob.checky(yval)
    print("check ytest")
    prob.checky(ytest)


    # Check a baseline first
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)

    ytrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(ytrainpred, ytrain)

    yvalpred = reg.predict(xval)
    valdl2st = prob.dec_loss(yvalpred, yval)

    traindltrue = prob.dec_loss(ytrain, ytrain)
    valdltrue = prob.dec_loss(yval, yval)
    testdltrue = prob.dec_loss(ytest, ytest)

    traindlrand = prob.rand_loss(ytrain)
    valdlrand = prob.rand_loss(yval)
    testdlrand = prob.rand_loss(ytest)

    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest)


    model = DirectedLoss(prob, params, xtrain, ytrain, xval, yval, valdltrue)
    scenario = Scenario(model.configspace, n_trials=args.n_trials)
    smac = HPOFacade(scenario, model.train, overwrite=True)
    incumbent = smac.optimize()


    weight_vec = model.get_vec(model)
    print(f"SMAC choose {weight_vec}")

    cusloss = search_weights_directed_loss(ytrain.shape[0], ytrain.shape[1], weight_vec)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train({"tree_method": params["tree_method"], "num_target": 2},
                             dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain)

    smacyvalpred = booster.inplace_predict(xval)
    valsmac = prob.dec_loss(smacyvalpred, yval)

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest)

    sanity_check(traindl2st - traindltrue, "train2st")
    sanity_check(valdl2st - valdltrue, "val2st")
    sanity_check(testdl2st - testdltrue, "test2st")
    sanity_check(trainsmac - traindltrue, "trainsmac")
    sanity_check(valsmac - valdltrue, "valsmac")
    sanity_check(testsmac - testdltrue, "testsmac")
    sanity_check(traindlrand - traindltrue, "trainrand")
    sanity_check(valdlrand - valdltrue, "valrand")
    sanity_check(testdlrand - testdltrue, "testrand")


    res_str= [(f"2stageTrainDL,2stageTrainDLstderr,2stageValDL,2stageValDLstderr,2stageTestDL,2stageTestDLstderr,"
               f"smacTrainDL,smacTrainDLsstderr,smacValDL,smacValDLstderr,smacTestDL,smacTestDLstderr,"
               f"randTrainDL,randTrainDLsstderr,randValDL,randValDLstderr,randTestDL,randTestDLstderr")]
    res_str.append((f"{(traindl2st - traindltrue).mean()}, {compute_stderror(traindl2st - traindltrue)}, "
                    f"{(valdl2st - valdltrue).mean()}, {compute_stderror(valdl2st - valdltrue)}, "
                    f"{(testdl2st - testdltrue).mean()}, {compute_stderror(testdl2st - testdltrue)},"
                    f"{(trainsmac - traindltrue).mean()}, {compute_stderror(trainsmac - traindltrue)}, "
                    f"{(valsmac - valdltrue).mean()}, {compute_stderror(valsmac - valdltrue)}, "
                    f"{(testsmac - testdltrue).mean()}, {compute_stderror(testsmac - testdltrue)},"
                    f"{(traindlrand - traindltrue).mean()}, {compute_stderror(traindlrand - traindltrue)}, "
                    f"{(valdlrand - valdltrue).mean()}, {compute_stderror(valdlrand - valdltrue)}, "
                    f"{(testdlrand - testdltrue).mean()}, {compute_stderror(testdlrand - testdltrue)}"))

    handcrapcusloss = search_weights_directed_loss(ytrain.shape[0], ytrain.shape[1], np.array([0.01, 100]))
    hcbooster = xgb.train({"tree_method": params["tree_method"], "num_target": 2},
                             dtrain = Xy, num_boost_round = params["search_estimators"], obj = handcrapcusloss.get_obj_fn())

    hctrainpred = hcbooster.inplace_predict(xtrain)
    hctrain = prob.dec_loss(hctrainpred, ytrain)

    hcvalpred = hcbooster.inplace_predict(xval)
    hcval = prob.dec_loss(hcvalpred, yval)

    hctestpred = hcbooster.inplace_predict(xtest)
    hctest = prob.dec_loss(hctestpred, ytest)

    res_str.append((f"Handcrafted,{(hctrain - traindltrue).mean()}, {compute_stderror(hctrain - traindltrue)}, "
                    f"{(hcval - valdltrue).mean()}, {compute_stderror(hcval - valdltrue)}, "
                    f"{(hctest - testdltrue).mean()}, {compute_stderror(hctest - testdltrue)}"))

    for row in res_str:
        print(row)

        #TODO how Lower L map to y0y1

