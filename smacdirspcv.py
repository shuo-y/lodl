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
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss, search_weights_loss, squared_error
from ShortestPath import ShortestPath
from smacdirected import DirectedLoss, QuadSearch, DirectedLossCrossValidation, WeightedLossCrossValidation, SearchbyInstanceCrossValid, XGBHyperSearch, DirectedLossCrossValHyper, QuadLossCrossValidation, test_config, test_config_vec, test_dir_weight, eval_xgb_hyper, contin_xgb_hyper
from utils import perfrandomdq, print_dq, print_nor_dq, compute_stderror, sanity_check, print_booster_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--search-method", type=str, default="wmse", choices=["wmse", "mse++", "idx", "msehyp++", "quad"])
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
    parser.add_argument("--xgb-lr", type=float, default=0.3, help="Used for xgboost eta")
    parser.add_argument("--n-test-history", type=int, default=0, help="Test history every what iterations default 0 not checking history")
    parser.add_argument("--cv-fold", type=int, default=5)
    parser.add_argument("--test-hyper", type=str, default="none", choices=["none", "hyperonly", "hyperwdef", "xgbregressorapi"], help="Ways of testing Hyperparameters")
    parser.add_argument("--test-nn2st", type=str, default="none", choices=["none", "dense", "dense_coupled"], help="Test nn two-stage model")
    parser.add_argument("--nn-lr", type=float, default=0.01, help="The learning rate for nn")
    parser.add_argument("--nn-iters", type=int, default=5000, help="Iterations for traning NN")
    parser.add_argument("--batchsize", type=int, default=1000, help="batchsize when traning NN")
    parser.add_argument("--n-layers", type=int, default=2, help="num of layers when traning NN") # What happens if n-layers much more than two?
    parser.add_argument("--int-size", type=int, default=500, help="num of layers when traning NN")
    parser.add_argument("--baseline", type=str, default="none", choices=["none", "lancer"])
    parser.add_argument("--contin-hyper", type=int, default=0)
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
    start_time = time.time()
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrainvalall, ytrainvalall)
    print(f"TIME train use XGBRegressor, {time.time() - start_time}. seconds")

    ytrainvalpred = reg.predict(xtrainvalall)
    trainvaldl2st = prob.dec_loss(ytrainvalpred, ytrainvalall).flatten()
    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest).flatten()

    trainvaldltrue = prob.dec_loss(ytrainvalall, ytrainvalall).flatten()
    testdltrue = prob.dec_loss(ytest, ytest).flatten()

    print(f"msexgbAPItrainval, {((ytrainvalpred - ytrainvalall) ** 2).mean()}")
    print(f"msexgbAPItest, {((ytestpred - ytest) ** 2).mean()}")

    # The shape of decision is the same as label Y
    trainvaldlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytrainvalall).float(), Y_aux=None, trials=params["n_rand_trials"]).numpy().flatten()
    testdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytest).float(), Y_aux=None, trials=params["n_rand_trials"]).numpy().flatten()

    search_map_cv = {"wmse": WeightedLossCrossValidation, "mse++": DirectedLossCrossValidation, "idx": SearchbyInstanceCrossValid, "msehyp++": DirectedLossCrossValHyper, "quad": QuadLossCrossValidation}

    search_model = search_map_cv[args.search_method]
    model = search_model(prob, params, xtrainvalall, ytrainvalall, args.param_low, args.param_upp, args.param_def, nfold=params["cv_fold"], eta=params["xgb_lr"])

    booster, bltrainvaldl, bltestdl = test_config_vec(params, prob, model.get_xgb_params(), model.get_def_loss_fn().get_obj_fn(), xtrainvalall, ytrainvalall, None, xtest, ytest, None, desc="defparams")

    print_dq([trainvaldl2st, testdl2st, bltrainvaldl, bltestdl], ["trainval2st", "test2st", "trainvalbl", "testbl1"], -1.0)
    print_nor_dq("trainvalnor", [trainvaldl2st, bltrainvaldl], ["trainval2st", "trainvalbl"], trainvaldlrand, trainvaldltrue)
    print_nor_dq("testnor", [testdl2st, bltestdl], ["test2st", "testbl"], testdlrand, testdltrue)

    if params["test_hyper"] != "none":
        _, hypertraindl, hypertestdl = eval_xgb_hyper(params, prob, xtrainvalall, ytrainvalall, None, xtest, ytest, None)
        print_dq([hypertraindl, hypertestdl], ["hypertrain","hypertest"], -1.0)
        print_nor_dq("trainvalnor", [hypertraindl], ["hypertrain"], trainvaldlrand, trainvaldltrue)
        print_nor_dq("testnor", [hypertestdl], ["hypertest"], testdlrand, testdltrue)
        exit(0)

    if params["test_nn2st"] != "none":
        from train_dense import nn2st_iter, perf_nn
        start_time = time.time()
        model = nn2st_iter(prob, xtrainvalall, ytrainvalall, None, None, params["nn_lr"], params["nn_iters"], params["batchsize"], params["n_layers"], params["int_size"], model_type=params["test_nn2st"], print_freq=(1+params["n_test_history"]))
        print(f"TIME train nn2st takes, {time.time() - start_time}, seconds")
        # Here just use the same data for tuning
        nntestdl = perf_nn(prob, model, xtest, ytest, None, desc="test")
        nntrainvaldl = perf_nn(prob, model, xtrainvalall, ytrainvalall, None, desc="trainval")

        print_dq([nntrainvaldl, nntestdl], ["NN2stTrainval", "NN2stTest"], -1.0)
        print_nor_dq("nn2stTrainvalNorDQ", [nntrainvaldl], ["NN2stTrainval"], trainvaldlrand, trainvaldltrue)
        print_nor_dq("nn2stTestNorDQ", [nntestdl], ["NN2stTestdl"], testdlrand, testdltrue)
        exit(0)

    if params["baseline"] == "lancer":
        from lancer_learner import test_lancer
        model, lctrainvaldl, lctestdl = test_lancer(prob, xtrainvalall, ytrainvalall, None, xtest, ytest, None,
                                                lancer_in_dim=prob.d, c_out_dim=prob.d, n_iter=10, c_max_iter=5, c_nbatch=128,
                                                lancer_max_iter=5, lancer_nbatch=1024, c_epochs_init=30, c_lr_init=0.005,
                                                lancer_lr=0.001, c_lr=0.005, lancer_n_layers=2, lancer_layer_size=100, c_n_layers=0, c_layer_size=64,
                                                lancer_weight_decay=0.01, c_weight_decay=0.01, z_regul=0.0,
                                                lancer_out_activation="relu", c_hidden_activation="tanh", c_output_activation="relu", print_freq=(1+params["n_test_history"]))

        print_dq([lctrainvaldl, lctestdl], ["LANCERtrainval", "LANCERtest"], -1.0)
        print_nor_dq("LANCERTrainvalNorDQ", [lctrainvaldl], ["LANCERTrainval"], trainvaldlrand, trainvaldltrue)
        print_nor_dq("LANCERTestNorDQ", [lctestdl], ["LANCERTestdl"], testdlrand, testdltrue)
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

        if params["n_test_history"] > 0 and cnt % params["n_test_history"] == 0:
            _, itertrainval, itertest = test_config_vec(params, prob, model.get_xgb_params(), model.get_loss_fn(info.config).get_obj_fn(), xtrainvalall, ytrainvalall, None, xtest, ytest, None)
            print(f"iter{cnt}: cost is {cost} config is {model.get_vec(info.config)}")
            print_dq([itertrainval, itertest], [f"iter{cnt}trainval", f"iter{cnt}test"], -1.0)
            print_nor_dq(f"iternordqtrainval", [itertrainval], [f"iter{cnt}trainval"], trainvaldlrand, trainvaldltrue)
            print_nor_dq(f"iternordqtest", [itertest], [f"iter{cnt}test"], testdlrand, testdltrue)
            # Check perf if stop now
            candidatesit = sorted(records, key=lambda x : x[0])
            select = 1
            for i in range(1, len(candidatesit)):
                if candidatesit[i][0] != candidatesit[0][0]:
                    select = i
                    print(f"iter SF{cnt}:from idx 0 to {select} has the same cost randomly pick one")
                    break
            idx = random.randint(0, select - 1)
            incumbent = candidatesit[idx][1]
            bstsf, itertrain, itertest = test_config_vec(params, prob, model.get_xgb_params(), model.get_loss_fn(incumbent).get_obj_fn(), xtrainvalall, ytrainvalall, None, xtest, ytest, None)
            print(f"SearchSoFar{cnt}: cost is {cost} config is {model.get_vec(info.config)}")
            print_dq([itertrain, itertest], [f"SF{cnt}train", f"SF{cnt}test"], -1.0)
            print_booster_mse(f"mseSFtrainval{cnt}", bstsf, xtrainvalall, ytrainvalall)
            print_booster_mse(f"mseSFtest{cnt}", bstsf, xtest, ytest)
            print("")
            print_nor_dq(f"SFnordqtrain", [itertrain], [f"SF{cnt}train"], trainvaldlrand, trainvaldltrue)
            print_nor_dq(f"SFnordqtest", [itertest], [f"SF{cnt}test"], testdlrand, testdltrue)


    candidates = sorted(records, key=lambda x : x[0])
    select = len(candidates) - 1
    for i in range(1, len(candidates)):
        if candidates[i][0] != candidates[0][0]:
            select = i
            print(f"from idx 0 to {select} has the same cost randomly pick one")
            break
    idx = random.randint(0, select - 1)
    incumbent = candidates[idx][1]
    print(f"TIME Search takes, {time.time() - start_time}, seconds")

    params_vec = model.get_vec(incumbent)
    print(f"print {incumbent}")
    print(f"Search Choose {params_vec}, val cost, {candidates[idx][0]},")

    start_time = time.time()
    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrainvalall, ytrainvalall)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())
    print(f"TIME Final train time, {time.time() - start_time}, seconds")

    print_booster_mse(f"mseFinaltrain", booster, xtrainvalall, ytrainvalall)
    print_booster_mse(f"mseFinaltest", booster, xtest, ytest)
    print("")

    print(f"Just in case: candidates[:5].mean {candidates[:5]} and records [:5].mean {records[:5]} ")

    smacytrainvalpred = booster.inplace_predict(xtrainvalall)
    trainvalsmac = prob.dec_loss(smacytrainvalpred, ytrainvalall).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest).flatten()

    _, bltrainvalfirst, bltestfirst = test_config_vec(params, prob, model.get_xgb_params(), model.get_loss_fn(records[0][1]).get_obj_fn(), xtrainvalall, ytrainvalall, None, xtest, ytest, None, desc="search1st") # Check the performance of the first iteration

    print_dq([trainvalsmac, testsmac, bltestdl, bltrainvalfirst, bltestfirst], ["trainvalsmac", "testsmac", "bldef", "bltrainvalfirst", "bltestfirst"], -1.0)
    print_nor_dq("Comparetrainvalnor", [trainvaldl2st, trainvalsmac], ["trainvaldl2st", "trainvalsmac"], trainvaldlrand, trainvaldltrue)
    print_nor_dq("Comparetestnor", [testdl2st, testsmac, bltestdl, bltestfirst], ["testdl2st", "testsmac", "bltestdl", "blfirst"], testdlrand, testdltrue)

    if params["contin_hyper"] > 0:
        _, hypertraindl, hypertestdl = contin_xgb_hyper(params, prob, xtrainvalall, ytrainvalall, None, xtest, ytest, None, cusloss.get_obj_fn(), params["contin_hyper"])
        print_dq([hypertraindl, hypertestdl], ["continhypertrain","continhypertest"], -1.0)
        print_nor_dq("con_trainvalnor", [hypertraindl], ["continhypertrain"], trainvaldlrand, trainvaldltrue)
        print_nor_dq("con_testnor", [hypertestdl], ["continhypertest"], testdlrand, testdltrue)
        exit(0)







