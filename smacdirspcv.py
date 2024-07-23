import random
import sys
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
from smacdirected import DirectedLoss, QuadSearch, DirectedLossCrossValidation, SearchbyInstanceCrossValid, XGBHyperSearch, test_config, test_config_vec, test_dir_weight, eval_xgb_hyper
from train_dense import nn2st_iter, perf_nn
from utils import perfrandomdq, print_dq, print_nor_dq, compute_stderror, sanity_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--search-method", type=str, default="mse++", choices=["mse++", "quad", "idx"])
    parser.add_argument("--search_estimators", type=int, default=100)
    parser.add_argument("--output", type=str, default="two_quad_example")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=400)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-rand-trials", type=int, default=10)
    parser.add_argument("--solver", type=str, choices=["scip", "gurobi", "glpk"], default="scip", help="optimization solver to use")
    parser.add_argument("--numfeatures", type=int, default=5)
    parser.add_argument("--spgrid", type=str, default="(5, 5)")
    parser.add_argument("--param-low", type=float, default=0.001)
    parser.add_argument("--param-upp", type=float, default=2.5)
    parser.add_argument("--param-def", type=float, default=0.05)
    parser.add_argument("--n-test-history", type=int, default=0, help="Test history every what iterations default 0 not checking history")
    parser.add_argument("--cv-fold", type=int, default=5)
    parser.add_argument("--test-hyper", type=str, default="none", choices=["none", "hyperonly", "hyperwdef"], help="Ways of testing Hyperparameters")
    parser.add_argument("--test-nn2st", type=str, default="none", choices=["none", "dense", "dense_coupled"], help="Test nn two-stage model")
    parser.add_argument("--lr", type=float, default=0.1, help="The learning rate for nn")
    parser.add_argument("--nn-iters", type=int, default=5000, help="Iterations for traning NN")
    parser.add_argument("--batchsize", type=int, default=1000, help="batchsize when traning NN")
    parser.add_argument("--n-layers", type=int, default=2, help="num of layers when traning NN") # What happens if n-layers much more than two?
    parser.add_argument("--int-size", type=int, default=500, help="num of layers when traning NN")
    #parser.add_argument("--cross-valid", action="store_true", help="Use cross validation during search")

    args = parser.parse_args()
    params = vars(args)

    # Load problem
    print(f"File:{sys.argv[0]} Hyperparameters: {args}\n")


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

    ytrainvalpred = reg.predict(xtrainvalall)
    trainvaldl2st = prob.dec_loss(ytrainvalpred, ytrainvalall).flatten()
    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest).flatten()

    trainvaldltrue = prob.dec_loss(ytrainvalall, ytrainvalall).flatten()
    testdltrue = prob.dec_loss(ytest, ytest).flatten()


    print(f"2st(trained on both train and val) trainval test val dl, {trainvaldl2st.mean()}, {compute_stderror(trainvaldl2st)}, "
          f"{testdl2st.mean()}, {compute_stderror(testdl2st)}, ")


    # The shape of decision is the same as label Y
    trainvaldlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytrainvalall).float(), Y_aux=None, trials=params["n_rand_trials"]).numpy().flatten()
    testdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytest).float(), Y_aux=None, trials=params["n_rand_trials"]).numpy().flatten()

    search_map = {"mse++": DirectedLoss, "quad": QuadSearch}
    search_map_cv = {"mse++": DirectedLossCrossValidation, "idx": SearchbyInstanceCrossValid}

    search_model = search_map_cv[args.search_method]
    model = search_model(prob, params, xtrainvalall, ytrainvalall, args.param_low, args.param_upp, args.param_def, nfold=params["cv_fold"])

    _, bltestdl = test_config_vec(params, prob, model.get_xgb_params(),  model.get_def_loss_fn(), xtrainvalall, ytrainvalall, xtest, ytest, None)
    print(f"Baseline:val train_val_all, {bltestdl.mean()}, {compute_stderror(bltestdl)}")

    print_dq([trainvaldl2st, testdl2st, bltestdl], ["trainval2st", "test2st", "bl1"], -1.0)
    print_nor_dq("trainvalnor", [trainvaldl2st], ["trainval2st"], trainvaldlrand, trainvaldltrue)
    print_nor_dq("testnor", [testdl2st, bltestdl], ["test2st", "bltest"], testdlrand, testdltrue)

    if params["test_hyper"] != "none":
        _, hypertestdl = eval_xgb_hyper(params, model.get_def_loss_fn(), prob, xtrainvalall, ytrainvalall, xtest, ytest, None)
        print(f"Hypertest dl: {hypertestdl.mean()}, {compute_stderror(hypertestdl)}")
        print_dq([hypertestdl], ["hypertest"], -1.0)
        print_nor_dq("testnor", [hypertestdl], ["hypertest"], testdlrand, testdltrue)
        exit(0)

    if params["test_nn2st"] != "none":
        model = nn2st_iter(prob, xtrainvalall, ytrainvalall, None, None, params["lr"], params["nn_iters"], params["batchsize"], params["n_layers"], params["int_size"], model_type=params["test_nn2st"])
        nntestdl = perf_nn(prob, model, xtest, ytest, None)
        print_dq([nntestdl], ["NNTestdl"], -1.0)
        print_nor_dq("nntestNorDQ", [nntestdl], ["NNTestdl"], testdlrand, testdltrue)
        exit(0)

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

        if cnt == 0:
            _, blfirst = test_config_vec(params, prob, model.get_xgb_params(),  model.get_loss_fn(info.config), xtrainvalall, ytrainvalall, xtest, ytest, None)

        if params["n_test_history"] > 0 and cnt % params["n_test_history"] == 0:
            _, trainvaldl, trainvaldlstderr = test_config(params, prob, model.get_xgb_params(),  model.get_loss_fn(info.config), xtrainvalall, ytrainvalall, xtest, ytest, None)
            print(f"Vec {model.get_vec(info.config)}")
            print(f"history iter{cnt} vol test teststderr, {cost}, {trainvaldl}, {trainvaldlstderr}")

    print(f"Search takes {time.time() - start_time} seconds")

    candidates = sorted(records, key=lambda x : x[0])
    select = len(candidates) - 1
    for i in range(1, len(candidates)):
        if candidates[i][0] != candidates[0][0]:
            select = i
            print(f"from idx 0 to {select} has the same cost randomly pick one")
            break
    idx = random.randint(0, select - 1)
    incumbent = records[idx][1]
    params_vec = model.get_vec(incumbent)
    print(f"Seaerch Choose {params_vec}")
    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrainvalall, ytrainvalall)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainvalpred = booster.inplace_predict(xtrainvalall)
    trainvalsmac = prob.dec_loss(smacytrainvalpred, ytrainvalall).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest).flatten()

    print_dq([trainvaldl2st, testdl2st, trainvalsmac, testsmac, bltestdl, blfirst], ["trainval2st", "test2st", "trainvalsmac", "testsmac", "bl1", "blfirstcnt"], -1.0)
    print_nor_dq("trainvalnor", [trainvaldl2st, trainvalsmac], ["trainvaldl2st", "trainvalsmac"], trainvaldlrand, trainvaldltrue)
    print_nor_dq("testnor", [testdl2st, testsmac, bltestdl, blfirst], ["testdl2st", "testsmac", "bltestdl", "blfirst"], testdlrand, testdltrue)






