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
from smacdirected import QuantileSearch

from PThenO import PThenO
from demoProb import ProdObj, optsingleprod, opttwoprod, gen_xy_twoprod
from utils import print_dq, print_nor_dq, print_nor_dq_filter0clip, print_booster_mse, print_multi_mse



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
    parser.add_argument("--search-estimators", type=int, default=32, help="num of estimaters during search")
    parser.add_argument("--final-estimators", type=int, default=100, help="num of estimaters at final train")
    parser.add_argument("--xgb-lr", type=float, default=0.3, help="Used for xgboost eta")
    parser.add_argument("--xgb-max-depth", type=int, default=10, help="max depth parameters for xgb")
    parser.add_argument("--xgb-earlystop", type=int, default=2, help="early stop for xgboost")

    parser.add_argument("--num-feats", type=int, default=10)
    parser.add_argument("--cov-mat", type=str, default="[[1, -0.9], [-0.9, 1]]")
    parser.add_argument("--gen-method", type=str, default="generate_dataset")
    parser.add_argument("--mus", type=str, default="[-0.1, -0.1]")
    parser.add_argument("--sigs", type=str, default="[0.1, 100]")
    parser.add_argument("--skewed-a", type=float, default=-5)

    parser.add_argument("--num-train", type=int, default=500)
    #parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--num-test", type=int, default=2000)
    parser.add_argument("--dnum", type=int, default=10, help="The number of d contained in each y instance")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--search-method", type=str, default="qt", choices=["qt"])
    #parser.add_argument("--power-scale", type=int, default=0)
    # The rand trials for perf rand dq
    parser.add_argument("--n-rand-trials", type=int, default=10)
    # For search and XGB train
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--param-low", type=float, default=0.05)
    parser.add_argument("--param-upp", type=float, default=0.95)
    parser.add_argument("--param-def", type=float, default=0.5)

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
    num_feat=args.num_feats
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

    ytrainpred2st = reg.predict(xtrain)
    traindecs, traindl2st = prob.dec_loss(ytrainpred2st, ytrain, return_dec=True)
    traindl2st = traindl2st.flatten()


    ytestpred2st = reg.predict(xtest)
    testdecs, testdl2st = prob.dec_loss(ytestpred2st, ytest, return_dec=True)
    testdl2st = testdl2st.flatten()

    truetraindc, traindltrue = prob.dec_loss(ytrain, ytrain, return_dec=True)
    truetestdc, testdltrue = prob.dec_loss(ytest, ytest, return_dec=True)

    traindlrand =  perf_rand(prob, ytrain, params["n_rand_trials"])  # TODO how to get the random decisions
    testdlrand = perf_rand(prob, ytest, params["n_rand_trials"])

    print_dq([traindltrue, testdltrue, traindl2st, testdl2st, traindlrand, testdlrand], ["traintrue", "testtrue","train2st", "test2st", "trainrand", "testrand"], -1.0)


    print_nor_dq("_trainnor", [traindl2st], ["train2st"], traindlrand, traindltrue)
    print_nor_dq("_testnor", [testdl2st], ["test2st"], testdlrand, testdltrue)
    print_nor_dq_filter0clip("trainnor2st", [traindl2st], ["train2st"], traindlrand, traindltrue)
    print_nor_dq_filter0clip("testnor2st", [testdl2st], ["test2st"], testdlrand, testdltrue)

    search_map_cv = {"qt": QuantileSearch}

    search_model = search_map_cv[params["search_method"]]
    model = search_model(prob, params, xtrain, ytrain, params["param_low"], params["param_upp"], params["param_def"])

    scenario = Scenario(model.configspace, n_trials=params["n_trials"])
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)
    smac = HPOFacade(scenario, model.train, intensifier=intensifier, overwrite=True)

    records = []
    start_time = time.time()

    def pick_incumbent(records, desc=""):
        if len(records) == 1:
            return records[0][1], records[0][2]
        candidates = sorted(records, key=lambda x : x[0])
        select = len(candidates) - 1
        for i in range(1, len(candidates)):
            if candidates[i][0] != candidates[0][0]:
                select = i
                print(f"from idx 0 to {select} has the same cost randomly pick one")
                break
        idx = random.randint(0, select - 1)
        incumbent = candidates[idx][1]
        models = candidates[idx][2]
        print(f"{desc}, choose val cost, {candidates[idx][0]},")
        return incumbent, models


    for cnt in range(params["n_trials"]):
        info = smac.ask()
        assert info.seed is not None

        cost, boosters = model.train(info.config, seed=info.seed, return_model=True)
        value = TrialValue(cost=cost)
        records.append((value.cost, info.config, boosters))
        smac.tell(info, value)

        if params["n_test_history"] > 0  and cnt % params["n_test_history"] == 0:
            # Check the Test DQ in this search iteration
            ytrainpred = model.pred(boosters, xtrain)
            traindliter = prob.dec_loss(ytrainpred, ytrain).flatten()

            ytestpred = model.pred(boosters, xtest)
            testdliter = prob.dec_loss(ytestpred, ytest).flatten()


            # Check the test DQ if stop now
            configs, sofarbests = pick_incumbent(records, desc=f"iter{cnt}_sofar")
            print(f"iter{cnt}_sofar, choose vec, {model.get_vec(configs)}")

            ytrainpredsf = model.pred(sofarbests, xtrain)
            traindlsf = prob.dec_loss(ytrainpredsf, ytrain).flatten()


            ytestpredsf = model.pred(sofarbests, xtest)
            testdlsf = prob.dec_loss(ytestpredsf, ytest).flatten()
            print(f"iter{cnt}, val cost, {cost}, traindliter, {traindliter.mean()}, testdliter, {testdliter.mean()}, traindlsf, {traindlsf.mean()}, testdlsf, {testdlsf.mean()}")

            print_multi_mse(f"MSEiter{cnt}", [ytrainpred, ytestpred, ytrainpredsf, ytestpredsf], [ytrain, ytest, ytrain, ytest], ["CurMSETrain", "CurMSETest", "SFMSETrain", "SFMSETest"])




    print(f"TIME Search takes, {time.time() - start_time}, seconds")
    incumbent, _finalmodels = pick_incumbent(records, desc="Finalconfigs")
    params_vec = model.get_vec(incumbent)
    print(f"print {incumbent}")
    print(f"Search Choose {params_vec}")


    start_time = time.time()
    fimodels = model.train(incumbent, seed=0, xtrain=xtrain, ytrain=ytrain, train_only=True)
    print(f"TIME Final train time, {time.time() - start_time}, seconds")

    ytrainpredfi =model.pred(fimodels, xtrain)
    trainsmac = prob.dec_loss(ytrainpredfi, ytrain).flatten()

    ytestpredfi = model.pred(fimodels, xtest)
    testsmac = prob.dec_loss(ytestpredfi, ytest).flatten()


    print_dq([trainsmac, testsmac], ["trainsmac", "testsmac"], -1.0)
    print_nor_dq("_Comparetrainnor", [traindl2st, trainsmac], ["traindl2st", "trainsmac"], traindlrand, traindltrue)
    print_nor_dq("_Comparetestnor", [testdl2st, testsmac], ["testdl2st", "testsmac"], testdlrand, testdltrue)
    print_nor_dq_filter0clip("Comparetrainnor", [traindl2st, trainsmac], ["traindl2st", "trainsmac"], traindlrand, traindltrue)
    print_nor_dq_filter0clip("Comparetestnor", [testdl2st, testsmac], ["testdl2st", "testsmac"], testdlrand, testdltrue)

    print_multi_mse(f"MSECompare", [ytrainpred2st, ytestpred2st, ytrainpredfi, ytestpredfi], [ytrain, ytest, ytrain, ytest], ["MSETrain2st", "MSETest2st", "FinalMSETrain", "FinalMSETest"])


