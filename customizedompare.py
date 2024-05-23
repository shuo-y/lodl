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
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss
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
    parser.add_argument("--search-estimators", type=int, default=100)
    parser.add_argument("--output", type=str, default="two_quad_example")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-heldval", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--start-time", type=str, default="dt.datetime(2004, 1, 1)")
    parser.add_argument("--end-time", type=str, default="dt.datetime(2017, 1, 1)")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--stocks", type=int, default=50)
    parser.add_argument("--stockalpha", type=float, default=1)
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

    total_num = args.num_train + args.num_val + args.num_heldval + args.num_test

    X = np.random.rand(total_num, 5)
    Y = np.random.rand(total_num, 10)

    indices = list(range(total_num))
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]]

    xval = X[indices[args.num_train:(args.num_train + args.num_val)]]
    yval = Y[indices[args.num_train:(args.num_train + args.num_val)]]

    xheld = X[indices[(args.num_train + args.num_val):(args.num_train + args.num_val + args.num_heldval)]]
    yheld = Y[indices[(args.num_train + args.num_val):(args.num_train + args.num_val + args.num_heldval)]]

    xtest = X[indices[(args.num_train + args.num_val + args.num_heldval):]]
    ytest = Y[indices[(args.num_train + args.num_val + args.num_heldval):]]


    # Check a baseline first
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)
    ypred2st = reg.predict(xtest)

    weight_vec = np.array([1.0 for _ in range(2 * ytrain.shape[1])])
    cusloss = search_weights_directed_loss(weight_vec)
    Xy = xgb.DMatrix(xtrain, ytrain)

    booster = xgb.train({"tree_method": params["tree_method"], "num_target": yval.shape[1]},
                            dtrain = Xy, num_boost_round = params["search_estimators"], xgb_model=reg.get_booster(), obj = cusloss.get_obj_fn())


    ypredcus = booster.inplace_predict(xtest)
    print(f"{((ypred2st - ytest) ** 2).mean()}, {((ypredcus - ytest) ** 2).mean()}")

