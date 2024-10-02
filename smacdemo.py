import random
import time
import argparse
import numpy as np
import torch
import xgboost as xgb


# From SMAC examples
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from losses import search_weights_loss, search_quadratic_loss

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from smacdirected import DirectedLossCrossValidation, WeightedLossCrossValidation, SearchbyInstanceCrossValid, DirectedLossCrossValHyper, QuadLossCrossValidation, test_config_vec, test_boosters, test_boosters_avepred, test_multi_reg

from PThenO import PThenO
from demoProb import ProdObj, optsingleprod, opttwoprod, gen_xy_twoprod
from utils import print_dq, print_nor_dq, print_nor_dq_filter0clip, print_booster_mse



def check_yins(y_arr, desc=""):
    negneg = 0
    negpos = 0
    posneg = 0
    pospos = 0
    zero0 = 0
    zero1 = 0
    for i in range(len(y_arr)):
        if y_arr[i][0] < 0 and y_arr[i][1] < 0:
            negneg += 1
        elif y_arr[i][0] < 0 and y_arr[i][1] >= 0:
            negpos += 1
        elif y_arr[i][0] >= 0  and y_arr[i][1] < 0:
            posneg += 1
        elif y_arr[i][0] >= 0 and y_arr[i][1] >= 0:
            pospos += 1
        if y_arr[i][0] == 0:
            zero0 += 1
        if y_arr[i][1] == 0:
            zero1 += 1

    print(f"{desc}, checkprop, --, {negneg}, -+, {negpos}, +-, {posneg}, ++, {pospos}, y0=0, {zero0}, y1=0, {zero1}")


def check_diff(y_pred, y_true, desc=""):
    assert len(y_pred) == len(y_true)
    matchsign = 0
    for i in range(len(y_pred)):
        if int(y_pred[i][0] > 0) == int(y_true[i][0] > 0) and int(y_pred[i][1] > 0) == int(y_true[i][1] > 0):
            matchsign += 1
    print(f"{desc}, checksign, {matchsign}. out of, {len(y_pred)}, sign matches")



def compute_stderror(vec: np.ndarray) -> float:
    popstd = vec.std()
    n = len(vec)
    return (popstd * np.sqrt(n / (n - 1.0))) / np.sqrt(n)

def sanity_check(vec: np.ndarray, msg: str) -> None:
    if (vec < 0).any():
        print(f"{msg}: check some negative value {vec}")


def perf_rand(prob, y_true, n_rand_trials):
    objs_rand = []
    for i in range(n_rand_trials):
        objs_rand.append(prob.dec_loss(np.random.random(y_true.shape), y_true))
    randomdqs = np.stack(objs_rand).mean(axis=0)
    return randomdqs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search_estimators", type=int, default=100)
    parser.add_argument("--num-feats", type=int, default=10)
    parser.add_argument("--cov-mat", type=str, default="[[1, -0.9], [-0.9, 1]]")
    parser.add_argument("--gen-method", type=str, default="generate_dataset")
    parser.add_argument("--mus", type=str, default="[-0.1, -0.1]")
    parser.add_argument("--sigs", type=str, default="[0.1, 100]")
    parser.add_argument("--skewed-a", type=float, default=-5)

    parser.add_argument("--loss", type=str, default="quad", choices=["mse", "quad"])
    parser.add_argument("--num-train", type=int, default=500)
    #parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--num-test", type=int, default=2000)
    parser.add_argument("--dnum", type=int, default=10, help="The number of d contained in each y instance")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--search-method", type=str, default="mse++", choices=["mse++", "quad", "idx", "wmse"])
    parser.add_argument("--power-scale", type=int, default=0)
    # The rand trials for perf rand dq
    parser.add_argument("--n-rand-trials", type=int, default=10)
    # For search and XGB train
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--param-low", type=float, default=0.1)
    parser.add_argument("--param-upp", type=float, default=100)
    parser.add_argument("--param-def", type=float, default=10)
    parser.add_argument("--xgb-lr", type=float, default=0.3, help="Used for xgboost eta")
    parser.add_argument("--n-test-history", type=int, default=0, help="Test history every what iterations default 0 not checking history")
    parser.add_argument("--cv-fold", type=int, default=5)
    parser.add_argument("--use-randcv", action="store_true", help="Train CV with each fold the same prob sampling")
    parser.add_argument("--rndcv-train-prob", type=float, default=0.0, help="The probability for sampling as train")
    # For NN
    parser.add_argument("--test-nn2st", type=str, default="none", choices=["none", "dense"], help="Test nn two-stage model")
    parser.add_argument("--nn-lr", type=float, default=0.01, help="The learning rate for nn")
    parser.add_argument("--nn-iters", type=int, default=500, help="Iterations for traning NN")
    parser.add_argument("--batchsize", type=int, default=1000, help="batchsize when traning NN")
    parser.add_argument("--n-layers", type=int, default=2, help="num of layers when traning NN") # What happens if n-layers much more than two?
    parser.add_argument("--int-size", type=int, default=500, help="num of layers when traning NN")
    # Other baseline
    parser.add_argument("--baseline", type=str, default="none", choices=["none", "lancer"])


    args = parser.parse_args()
    params = vars(args)

    # Load problem
    print(f"Hyperparameters: {args}\n")


    random.seed(args.seed)
    np.random.seed(args.seed * 1091)
    torch.manual_seed(args.seed * 2621)

    params = vars(args)

    def fun1(x):
        return (0, 0.55)

    def fun2(x):
        return (1, 0.55)

    def rand_fun(x):
        if np.random.rand() >= 0.5:
            return (0, 0.55)
        return (1, 0.55)

    def quad_fun(x):
        # Similar to SPO demo
        y0 = 0.5 * (x ** 2) - 0.1
        y1 = 0.2 * (x ** 2)
        return (y0, y1)


    numInstance = args.num_train + args.num_test
    prob = ProdObj(opttwoprod, args.dnum)
    num_feat=10
    X, Y = gen_xy_twoprod(numInstance, args.dnum, 1000, 0.1, num_feat=num_feat)


    indices = list(range(args.num_train + args.num_test))
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]]


    xtest = X[indices[(args.num_train):]]
    ytest = Y[indices[(args.num_train):]]

    # This reshape is for XGBoost
    xtrain = xtrain.reshape(xtrain.shape[0] * xtrain.shape[1], num_feat)
    ytrain = ytrain.reshape(ytrain.shape[0] * ytrain.shape[1], 4)

    xtest = xtest.reshape(xtest.shape[0] * xtest.shape[1], num_feat)
    ytest = ytest.reshape(ytest.shape[0] * ytest.shape[1], 4)


    # Check a baseline first
    start_time = time.time()
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)
    print(f"TIME train use XGBRegressor, {time.time() - start_time} , seconds.")

    ytrainpred = reg.predict(xtrain)
    traindecs, traindl2st = prob.dec_loss(ytrainpred, ytrain, return_dec=True)
    traindl2st = traindl2st.flatten()

    ytestpred = reg.predict(xtest)
    testdecs, testdl2st = prob.dec_loss(ytestpred, ytest, return_dec=True)
    testdl2st = testdl2st.flatten()
    #check_diff(ytestpred, ytest, "checkdiff2st")
    #check_yins(ytestpred, "check_yins_2stpred")
    #print("check ypred 2st,", end="")
    #prob.checky(ytestpred)

    truetraindc, traindltrue = prob.dec_loss(ytrain, ytrain, return_dec=True)
    truetestdc, testdltrue = prob.dec_loss(ytest, ytest, return_dec=True)
    print_dq([traindltrue, testdltrue, traindl2st, testdl2st], ["traintrue", "testtrue","train2st", "test2st"], -1.0)

    traindlrand =  perf_rand(prob, ytrain, params["n_rand_trials"])
    testdlrand = perf_rand(prob, ytest, params["n_rand_trials"])

    print_nor_dq("_trainnor", [traindl2st], ["train2st"], traindlrand, traindltrue)
    print_nor_dq("_testnor", [testdl2st], ["test2st"], testdlrand, testdltrue)
    print_nor_dq_filter0clip("trainnor2st", [traindl2st], ["train2st"], traindlrand, traindltrue)
    print_nor_dq_filter0clip("testnor2st", [testdl2st], ["test2st"], testdlrand, testdltrue)


    search_map_cv = {"wmse": WeightedLossCrossValidation, "mse++": DirectedLossCrossValidation, "idx": SearchbyInstanceCrossValid, "msehyp++": DirectedLossCrossValHyper, "quad": QuadLossCrossValidation}

    search_model = search_map_cv[params["search_method"]]
    model = search_model(prob, params, xtrain, ytrain, params["param_low"], params["param_upp"], params["param_def"], nfold=params["cv_fold"], eta=params["xgb_lr"], use_rand_cv=params["use_randcv"], prob_train=params["rndcv_train_prob"], power_scale=params["power_scale"])

    booster, bltraindl, bltestdl = test_config_vec(params, prob, model.get_xgb_params(), model.get_def_loss_fn().get_obj_fn(), xtrain, ytrain, None, xtest, ytest, None, desc="defparams")
    print_dq([bltraindl, bltestdl], ["bltraindl", "bltestdl"], -1.0)
    print_nor_dq("_trainnor", [bltraindl], ["bltraindl"], traindlrand, traindltrue)
    print_nor_dq("_testnor", [bltestdl], ["bltestdl"], testdlrand, testdltrue)
    print_nor_dq_filter0clip("trainnorbl", [bltraindl], ["bltraindl"], traindlrand, traindltrue)
    print_nor_dq_filter0clip("testnorbl", [bltestdl], ["bltestdl"], testdlrand, testdltrue)


    if params["test_nn2st"] != "none":
        from train_dense import nn2st_iter, perf_nn
        start_time = time.time()
        model = nn2st_iter(prob, xtrain, ytrain, None, None, params["nn_lr"], params["nn_iters"], params["batchsize"], params["n_layers"], params["int_size"], model_type=params["test_nn2st"], print_freq=(1+params["n_test_history"]))
        print(f"TIME train nn2st takes, {time.time() - start_time}, seconds")
        # Here just use the same data for tuning
        nntestdl = perf_nn(prob, model, xtest, ytest, None)
        nntraindl = perf_nn(prob, model, xtrain, ytrain, None)

        print_dq([nntraindl, nntestdl], ["NN2stTrainval", "NN2stTest"], -1.0)
        print_nor_dq("_nn2stTrainNorDQ", [nntraindl], ["NN2stTrain"], traindlrand, traindltrue)
        print_nor_dq("_nn2stTestNorDQ", [nntestdl], ["NN2stTestdl"], testdlrand, testdltrue)
        print_nor_dq_filter0clip("nn2stTrainNorDQ", [nntraindl], ["NN2stTrain"], traindlrand, traindltrue)
        print_nor_dq_filter0clip("nn2stTestNorDQ", [nntestdl], ["NN2stTestdl"], testdlrand, testdltrue)

        exit(0)

    if params["baseline"] == "lancer":
        from lancer_learner import test_lancer
        model, lctraindl, lctestdl = test_lancer(prob, xtrain, ytrain, None, xtest, ytest, None,
                                                lancer_in_dim=2, c_out_dim=2, n_iter=10, c_max_iter=5, c_nbatch=128,
                                                lancer_max_iter=5, lancer_nbatch=1024, c_epochs_init=50, c_lr_init=0.005,
                                                lancer_lr=0.05, c_lr=0.005, lancer_n_layers=2, lancer_layer_size=100, c_n_layers=0, c_layer_size=64,
                                                lancer_weight_decay=0.01, c_weight_decay=0.01, z_regul=0.0,
                                                lancer_out_activation="relu", c_hidden_activation="tanh", c_output_activation="relu", print_freq=(1+params["n_test_history"]))

        print_dq([lctraindl, lctestdl], ["LANCERtrain", "LANCERtest"], -1.0)
        print_nor_dq("_LANCERTrainNorDQ", [lctraindl], ["LANCERTrain"], traindlrand, traindltrue)
        print_nor_dq("_LANCERTestNorDQ", [lctestdl], ["LANCERTestdl"], testdlrand, testdltrue)
        print_nor_dq_filter0clip("LANCERTrainNorDQ", [lctraindl], ["LANCERTrain"], traindlrand, traindltrue)
        print_nor_dq_filter0clip("LANCERTestNorDQ", [lctestdl], ["LANCERTestdl"], testdlrand, testdltrue)

        exit(0)

    scenario = Scenario(model.configspace, n_trials=params["n_trials"])
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)
    smac = HPOFacade(scenario, model.train, intensifier=intensifier, overwrite=True)

    records = []
    start_time = time.time()

    for cnt in range(params["n_trials"]):
        info = smac.ask()
        assert info.seed is not None

        if params["n_test_history"] > 0 and cnt % params["n_test_history"] == 0:
            cost, boosters = model.train(info.config, seed=info.seed, return_model=True)
        else:
            cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost)
        records.append((value.cost, info.config))
        smac.tell(info, value)

        if params["n_test_history"] > 0 and cnt % params["n_test_history"] == 0:
            bstavedltest = test_boosters(params, prob, boosters, xtest, ytest, None)
            bstavedltestavepred = test_boosters_avepred(params, prob, boosters, xtest, ytest, None)

            testbst, itertrain, itertest = test_config_vec(params, prob, model.get_xgb_params(), model.get_loss_fn(info.config).get_obj_fn(), xtrain, ytrain, None, xtest, ytest, None)
            ytestpred = testbst.inplace_predict(xtest)
            # Check value and itertest correlation
            params_vec = model.get_vec(info.config)
            print(f"smac sample,{info.config}")
            print(f"iter{cnt}, val cost is, {cost}, bsts ave dq, {bstavedltest.mean()}, bsts dq ave ypred, {bstavedltestavepred.mean()}, test cost, {itertest.mean()}, l1reg, {(abs(params_vec)).mean()}, l2reg, {(params_vec ** 2).mean()},config is, {model.get_vec(info.config)[0]}, {model.get_vec(info.config)[1]}, ypred is, {ytestpred[:,0].mean()}, {ytestpred[:,1].mean()}")
            print_dq([itertrain, itertest], [f"iter{cnt}train", f"iter{cnt}test"], -1.0)
            print_nor_dq(f"_iternordqtrain", [itertrain], [f"iter{cnt}train"], traindlrand, traindltrue)
            print_nor_dq(f"_iternordqtest", [itertest], [f"iter{cnt}test"], testdlrand, testdltrue)
            print_nor_dq_filter0clip(f"iternordqtrain", [itertrain], [f"iter{cnt}train"], traindlrand, traindltrue)
            print_nor_dq_filter0clip(f"iternordqtest", [itertest], [f"iter{cnt}test"], testdlrand, testdltrue)
            # Check so far
            candidatesit = sorted(records, key=lambda x : x[0])
            select = 1
            for i in range(1, len(candidatesit)):
                if candidatesit[i][0] != candidatesit[0][0]:
                    select = i
                    print(f"iter SF{cnt}:from idx 0 to {select} has the same cost randomly pick one")
                    break
            idx = random.randint(0, select - 1)
            incumbent = candidatesit[idx][1]
            bstsf, itertrain, itertest = test_config_vec(params, prob, model.get_xgb_params(), model.get_loss_fn(incumbent).get_obj_fn(), xtrain, ytrain, None, xtest, ytest, None)
            print(f"SearchSoFar{cnt}: cost is {cost} config is {model.get_vec(info.config)}")
            print_dq([itertrain, itertest], [f"SF{cnt}train", f"SF{cnt}test"], -1.0)
            print_booster_mse(f"mseSFtrain{cnt}", bstsf, xtrain, ytrain)
            print_booster_mse(f"mseSFtest{cnt}", bstsf, xtest, ytest)
            print("")
            print_nor_dq(f"SFnordqtrain", [itertrain], [f"SF{cnt}train"], traindlrand, traindltrue)
            print_nor_dq(f"SFnordqtest", [itertest], [f"SF{cnt}test"], testdlrand, testdltrue)
            print_nor_dq_filter0clip(f"SFnordqtrain", [itertrain], [f"SF{cnt}train"], traindlrand, traindltrue)
            print_nor_dq_filter0clip(f"SFnordqtest", [itertest], [f"SF{cnt}test"], testdlrand, testdltrue)


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
    print(f"Search Choose {params_vec}")
    print(f"Finalvalcost, {candidates[idx][0]}, l1reg, {(abs(params_vec)).mean()}, l2reg, {(params_vec ** 2).mean()}")

    cost, boosters = model.train(candidates[idx][1], return_model=True, seed=0)
    bstavedltrain = test_boosters(params, prob, boosters, xtrain, ytrain, None)
    bstavedltest = test_boosters(params, prob, boosters, xtest, ytest, None)

    bstaveystest = test_boosters_avepred(params, prob, boosters, xtest, ytest, None)

    start_time = time.time()
    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())
    print(f"TIME Final train time, {time.time() - start_time}, seconds")

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest).flatten()
    print("check ypred smac,", end="")
    prob.checky(smacytestpred)
    check_yins(smacytestpred, "check_yins_smacpred")
    check_diff(smacytestpred, ytest, "smacdiff")

    _, bltrainfirst, bltestfirst = test_config_vec(params, prob, model.get_xgb_params(), model.get_loss_fn(records[0][1]).get_obj_fn(), xtrain, ytrain, None, xtest, ytest, None, desc="search1st") # Check the performance of the first iteration

    twocostsavedqs, twotestaveysdl, twovalcosts = test_multi_reg(params, prob, model, xtest, ytest, None)

    print_dq([trainsmac, testsmac, bltestdl, bltrainfirst, bltestfirst, bstavedltrain, bstavedltest, bstaveystest, twocostsavedqs, twotestaveysdl, twovalcosts], ["trainsmac", "testsmac", "bldef", "bltrainfirst", "bltestfirst", "bstavedltrain", "bstavedltest", "bstaveystest", "twocostsavedqs", "twotestaveysdl", "twovalcosts"], -1.0)
    print_nor_dq("_Comparetrainnor", [traindl2st, trainsmac, bstavedltrain], ["traindl2st", "trainsmac", "bstavedltrain"], traindlrand, traindltrue)
    print_nor_dq("_Comparetestnor", [testdl2st, testsmac, bltestdl, bltestfirst, bstavedltest, bstaveystest, twocostsavedqs, twotestaveysdl], ["testdl2st", "testsmac", "bltestdl", "blfirst", "bstavedltest", "bstaveystest", "twocostsavedqs", "twotestaveysdl"], testdlrand, testdltrue)
    print_nor_dq_filter0clip("Comparetrainnor", [traindl2st, trainsmac, bstavedltrain], ["traindl2st", "trainsmac", "bstavedltrain"], traindlrand, traindltrue)
    print_nor_dq_filter0clip("Comparetestnor", [testdl2st, testsmac, bltestdl, bltestfirst, bstavedltest, bstaveystest, twocostsavedqs, twotestaveysdl], ["testdl2st", "testsmac", "bltestdl", "blfirst", "bstavedltest", "bstaveystest", "twocostsavedqs", "twotestaveysdl"], testdlrand, testdltrue)

    from smacdirected import check_val_and_test
    valfoldcost, testfoldcost = check_val_and_test(prob, model.params, model.Xys, model.valdatas, None, (xtest, ytest), None,
                                                  model.ydim, params_vec, model.nfold, 0)

    print(f"Corrcoef, Per Instance between test obj (train + val) and test obj (train only), {np.corrcoef(bstavedltest, testsmac)[0][1]},")
    print(f"Corrcoef, Per fold between val obj (train only) and test obj (train only), {np.corrcoef(valfoldcost, testfoldcost)[0][1]}, ")





