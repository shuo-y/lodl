import random
import sys
import argparse
import time
import numpy as np
import torch
import datetime as dt
import xgboost as xgb

"""
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss
from smacdirected import DirectedLoss, QuadSearch, DirectedLossCrossValidation, WeightedLossCrossValidation, SearchbyInstanceCrossValid, XGBHyperSearch, DirectedLossCrossValHyper, QuadLossCrossValidation, test_config, test_config_vec, test_dir_weight, test_reg, test_weightmse, test_square_log
from smacdirected import smac_search_lgb, eval_config_lgb, test_reg_lgb
"""
from utils import perfrandomdq, print_dq, print_nor_dq_filter0clip, print_nor_dqagg, compute_stderror, sanity_check
from BudgetAllocation import BudgetAllocation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--search-method", type=str, default="mse++", choices=["mse++", "quad", "idx", "wmse"])
    parser.add_argument("--search_estimators", type=int, default=100)
    parser.add_argument("--num-train", type=int, default=80)
    parser.add_argument("--num-val", type=int, default=20)
    parser.add_argument("--num-test", type=int, default=500)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-rand-trials", type=int, default=10)
    parser.add_argument("--n-targets", type=int, default=10, help="used for budget allocation number of users")
    parser.add_argument("--n-items", type=int, default=5, help="used for budget allocation number of websites")
    parser.add_argument("--n-budget", type=int, default=2, help="used for budget allocation budget number")
    parser.add_argument("--n-fake-targets", type=int, default=500, help="To make budget allocation harder")


    args = parser.parse_args()
    params = vars(args)

    # Load problem
    print(f"File:{sys.argv[0]} Hyperparameters: {args}\n")


    random.seed(args.seed)
    np.random.seed(args.seed * 1091)
    torch.manual_seed(args.seed * 2621)

    params = vars(args)

    prob = BudgetAllocation(num_train_instances=params["num_train"],  # number of instances to use from the dataset to train
        num_val_instances=params["num_val"],
        num_test_instances=params["num_test"],  # number of instances to use from the dataset to test
        num_targets=params["n_targets"],  # number of items to choose from
        num_items=params["n_items"],  # number of targets to consider
        budget=params["n_budget"],  # number of items that can be picked
        num_fake_targets=params["n_fake_targets"])

    Xtraint, Ytraint, _ = prob.get_train_data()
    Xtestt, Ytestt, _ = prob.get_test_data()

    xtrain = Xtraint.cpu().detach().numpy()
    ytrain = Ytraint.cpu().detach().numpy()

    xtest = Xtestt.cpu().detach().numpy()
    ytest = Ytestt.cpu().detach().numpy()

    xtrain = xtrain.reshape(Xtraint.shape[0], Xtraint.shape[1] * Xtraint.shape[2])
    ytrain = ytrain.reshape(Ytraint.shape[0], Ytraint.shape[1] * Ytraint.shape[2])

    xtest = xtest.reshape(Xtestt.shape[0], Xtestt.shape[1] * Xtestt.shape[2])
    ytest = ytest.reshape(Ytestt.shape[0], Ytestt.shape[1] * Ytestt.shape[2])

    # Check a baseline first
    start_time = time.time()
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)
    print(f"TIME train use XGBRegressor, {time.time() - start_time}. seconds")

    ytrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(ytrainpred, ytrain).flatten()

    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest).flatten()

    traindltrue = prob.dec_loss(ytrain, ytrain).flatten()
    testdltrue = prob.dec_loss(ytest, ytest).flatten()

    # The shape of decision is the same as label Y
    traindlrand = -1.0 * perfrandomdq(prob, Y=Ytraint, Y_aux=None, trials=params["n_rand_trials"]).numpy().flatten()
    testdlrand = -1.0 * perfrandomdq(prob, Y=Ytestt, Y_aux=None, trials=params["n_rand_trials"]).numpy().flatten()

    # TODO here some is 0 how to benchmark
    print_dq([traindl2st, testdl2st], ["train2st", "test2st"], -1.0)
    vbtrain = print_nor_dq_filter0clip("trainnor", [traindl2st], ["train2st"], traindlrand, traindltrue)
    vbtest = print_nor_dq_filter0clip("testnor", [testdl2st], ["test2st"], testdlrand, testdltrue)

    print_nor_dqagg("trainnor", [traindl2st], ["train2st"], traindlrand, traindltrue)
    print_nor_dqagg("testnor", [testdl2st], ["test2st"], testdlrand, testdltrue)


    search_map_cv = {"wmse": WeightedLossCrossValidation, "mse++": DirectedLossCrossValidation, "idx": SearchbyInstanceCrossValid, "msehyp++": DirectedLossCrossValHyper, "quad": QuadLossCrossValidation}

    search_model = search_map_cv[params["search_method"]]
    model = search_model(prob, params, xtrain, ytrain, args.param_low, args.param_upp, args.param_def, nfold=params["cv_fold"], eta=params["xgb_lr"])

    booster, bltraindl, bltestdl = test_config_vec(params, prob, model.get_xgb_params(), model.get_def_loss_fn().get_obj_fn(), xtrain, ytrain, None, xtest, ytest, None, desc="defparams")

    print_dq([traindl2st, testdl2st, bltraindl, bltestdl], ["train2st", "test2st", "trainbl", "testbl"], -1.0)
    print_nor_dq_filter0("trainnor", [traindl2st, bltraindl], ["train2st", "trainbl"], traindlrand, traindltrue)
    print_nor_dq_filter0("testnor", [testdl2st, bltestdl], ["test2st", "testbl"], testdlrand, testdltrue)

