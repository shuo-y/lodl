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
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss
from PortfolioOpt import PortfolioOpt
from smacdirected import DirectedLoss, QuadSearch, DirectedLossCrossValidation, SearchbyInstance, SearchbyInstanceCrossValid, test_config, test_config_vec, test_dir_weight, test_reg, test_weightmse, test_square_log
from smacdirected import smac_search_lgb, eval_config_lgb, test_reg_lgb
from utils import perfrandomdq, print_train_test, print_train_val_test, compute_stderror, sanity_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--search-method", type=str, default="mse++", choices=["mse++", "quad", "idx"])
    parser.add_argument("--search_estimators", type=int, default=100)
    parser.add_argument("--output", type=str, default="two_quad_example")
    parser.add_argument("--num-train", type=int, default=150)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--num-test", type=int, default=400)
    parser.add_argument("--start-time", type=str, default="dt.datetime(2004, 1, 1)")
    parser.add_argument("--end-time", type=str, default="dt.datetime(2017, 1, 1)")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--stocks", type=int, default=50)
    parser.add_argument("--stockalpha", type=float, default=0.1)
    parser.add_argument("--param-low", type=float, default=0.008)
    parser.add_argument("--param-upp", type=float, default=2)
    parser.add_argument("--param-def", type=float, default=0.04)
    parser.add_argument("--n-test-history", type=int, default=0, help="Check test dl of the history")
    parser.add_argument("--cross-valid", action="store_true", help="Use cross validation during search")
    parser.add_argument("--cv-fold", type=int, default=5)

    args = parser.parse_args()
    params = vars(args)

    # Load problem
    print(f"File:{sys.argv[0]} Hyperparameters: {args}\n")


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

    xtrain = X[indices[:params["num_train"]]]
    ytrain = Y[indices[:params["num_train"]]]
    auxtrain = Aux[indices[:params["num_train"]]]

    xval = X[indices[params["num_train"] : (params["num_train"] + params["num_val"])]]
    yval = Y[indices[params["num_train"] : (params["num_train"] + params["num_val"])]]
    auxval = Aux[indices[params["num_train"] : (params["num_train"] + params["num_val"])]]

    xtrainvalall = X[indices[:(params["num_train"] + params["num_val"])]]
    ytrainvalall = Y[indices[:(params["num_train"] + params["num_val"])]]
    auxtrainvalall = Aux[indices[:(params["num_train"] + params["num_val"])]]

    xtest = X[indices[(args.num_train + args.num_val):]]
    ytest = Y[indices[(args.num_train + args.num_val):]]
    auxtest = Aux[indices[(args.num_train + args.num_val):]]


    # Check a baseline first

    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    if params["cross_valid"] == True:
        reg.fit(xtrainvalall, ytrainvalall)
    else:
        reg.fit(xtrain, ytrain)

    ytrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(ytrainpred, ytrain, aux_data=auxtrain).flatten()

    yvalpred = reg.predict(xval)
    valdl2st = prob.dec_loss(yvalpred, yval, aux_data=auxval).flatten()

    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest, aux_data=auxtest).flatten()

    ytrainvalpred = reg.predict(xtrainvalall)
    trainvaldl2st = prob.dec_loss(ytrainvalpred, ytrainvalall, aux_data=auxtrainvalall).flatten()

    traindltrue = prob.dec_loss(ytrain, ytrain, aux_data=auxtrain).flatten()
    valdltrue = prob.dec_loss(yval, yval, aux_data=auxval).flatten()
    trainvaldltrue = prob.dec_loss(ytrainvalall, ytrainvalall, aux_data=auxtrainvalall).flatten()
    testdltrue = prob.dec_loss(ytest, ytest, aux_data=auxtest).flatten()


    print(f"2st(trained on both train and val) train test val obj, {traindl2st.mean()}, {compute_stderror(traindl2st)}, "
          f"{testdl2st.mean()}, {compute_stderror(testdl2st)}, "
          f"{valdl2st.mean()}, {compute_stderror(valdl2st)}, ")


    # The shape of decision is the same as label Y


    search_map = {"mse++": DirectedLoss, "quad": QuadSearch, "idx": SearchbyInstance}
    search_map_cv = {"mse++": DirectedLossCrossValidation, "idx": SearchbyInstanceCrossValid}


    if params["cross_valid"] == True:
          search_model = search_map_cv[args.search_method]
          model = search_model(prob, params, xtrainvalall, ytrainvalall, args.param_low, args.param_upp, args.param_def, auxdata=auxtrainvalall, nfold=params["cv_fold"])
          _, bltestdl = test_config_vec(params, prob, model.get_xgb_params(),  model.get_def_loss_fn(), xtrainvalall, ytrainvalall, xtest, ytest, auxtest)
          print(f"Baseline:val train_val_all, {bltestdl.mean()}, {compute_stderror(bltestdl)}")
    else:
          search_model = search_map[args.search_method]
          model = search_model(prob, params, xtrain, ytrain, xval, yval, args.param_low, args.param_upp, args.param_def, auxdata=auxval)

    #chosen_loss = smac_search_lgb(params, prob, model, params["n_trials"], xtrain, ytrain, xtest, ytest, auxtest)
    #lgb_model, trainsmacpred, valsmacpred, testsmacpred = eval_config_lgb(params, prob, xgb_params, chosen_loss, xtrain, ytrain, xval, xtest)


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
            print(f"Vec {model.get_vec(info.config)}")
            if params["cross_valid"] == True:
                  _, trainvaldl, trainvaldlstderr = test_config(params, prob, model.get_xgb_params(),  model.get_loss_fn(info.config), xtrainvalall, ytrainvalall, xtest, ytest, auxtest)
                  print(f"history:val train_val_all, {cost}, {trainvaldl}, {trainvaldlstderr}")
            else:
                _, testdl, teststderr = test_config(params, prob, model.get_xgb_params(), model.get_loss_fn(info.config), xtrain, ytrain, xtest, ytest, auxtest)
                print(f"history val test teststderr, {cost}, {testdl}, {teststderr}")

    print(f"Search takes {time.time() - start_time} seconds")

    records.sort()
    incumbent = records[0][1]
    params_vec = model.get_vec(incumbent)
    print(f"Seaerch Choose {params_vec}")
    cusloss = model.get_loss_fn(incumbent)
    if params["cross_valid"] == True:
        Xy = xgb.DMatrix(xtrainvalall, ytrainvalall)
    else:
        Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain, aux_data=auxtrain).flatten()

    smacyvalpred = booster.inplace_predict(xval)
    valsmac = prob.dec_loss(smacyvalpred, yval, aux_data=auxval).flatten()

    smactrainvalpred = booster.inplace_predict(xtrainvalall)
    trainvalsmac = prob.dec_loss(smactrainvalpred, ytrainvalall, aux_data=auxtrainvalall).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest, aux_data=auxtest).flatten()

    testdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytest), Y_aux=torch.tensor(auxtest), trials=10).numpy().flatten()

    if params["cross_valid"] == True:
        trainvaldlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytrainvalall), Y_aux=torch.tensor(auxtrainvalall), trials=10).numpy().flatten()
        print_train_test(trainvaldl2st, testdl2st, trainvalsmac, testsmac, trainvaldlrand, testdlrand, trainvaldltrue, testdltrue, bltestdl)
    else:
        traindlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(ytrain), Y_aux=torch.tensor(auxtrain), trials=10).numpy().flatten()
        valdlrand = -1.0 * perfrandomdq(prob, Y=torch.tensor(yval), Y_aux=torch.tensor(auxval), trials=10).numpy().flatten()
        print_train_val_test(traindl2st, valdl2st, testdl2st, trainsmac, valsmac, testsmac, traindlrand, valdlrand, testdlrand, traindltrue, valdltrue, testdltrue, bltestdl)




