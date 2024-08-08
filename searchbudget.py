import random
import sys
import argparse
import time
import numpy as np
import torch
import datetime as dt
import xgboost as xgb


from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss
from smacdirected import DirectedLoss, QuadSearch, DirectedLossCrossValidation, WeightedLossCrossValidation, SearchbyInstanceCrossValid, XGBHyperSearch, DirectedLossCrossValHyper, QuadLossCrossValidation, test_config, test_config_vec, test_dir_weight, test_reg, test_weightmse, test_square_log
from smacdirected import smac_search_lgb, eval_config_lgb, test_reg_lgb


from utils import perfrandomdq, print_dq, print_nor_dq_filter0clip, print_nor_dqagg, print_booster_mse, compute_stderror, sanity_check
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

    # For search and XGB train
    parser.add_argument("--param-low", type=float, default=0.008)
    parser.add_argument("--param-upp", type=float, default=2.0)
    parser.add_argument("--param-def", type=float, default=0.04)
    parser.add_argument("--xgb-lr", type=float, default=0.3, help="Used for xgboost eta")
    parser.add_argument("--n-test-history", type=int, default=0, help="Test history every what iterations default 0 not checking history")
    parser.add_argument("--cv-fold", type=int, default=5)
    parser.add_argument("--use-decouple", action="store_true", help="If use a decoupled version budget")
    # For NN
    parser.add_argument("--test-nn2st", type=str, default="none", choices=["none", "dense"], help="Test nn two-stage model")
    parser.add_argument("--nn-lr", type=float, default=0.00001, help="The learning rate for nn")
    parser.add_argument("--nn-iters", type=int, default=10000, help="Iterations for traning NN")
    parser.add_argument("--batchsize", type=int, default=1000, help="batchsize when traning NN")
    parser.add_argument("--n-layers", type=int, default=2, help="num of layers when traning NN") # What happens if n-layers much more than two?
    parser.add_argument("--int-size", type=int, default=500, help="num of layers when traning NN")
    # Other baseline
    parser.add_argument("--baseline", type=str, default="none", choices=["none", "lancer"])

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

    if params["use_decouple"]:
        print("Trained with decouple way")
        # Train with de couple fashion
        xtrain = xtrain.reshape(Xtraint.shape[0] * Xtraint.shape[1], Xtraint.shape[2])
        ytrain = ytrain.reshape(Ytraint.shape[0] * Ytraint.shape[1], Ytraint.shape[2])

        xtest = xtest.reshape(Xtestt.shape[0] * Xtestt.shape[1], Xtestt.shape[2])
        ytest = ytest.reshape(Ytestt.shape[0] * Ytestt.shape[1], Ytestt.shape[2])

        prob.use_decouple = True
        prob.num_feats = prob.num_targets
        prob.ipdim = prob.num_feats
        prob.opdim = prob.num_feats
    else:
        xtrain = xtrain.reshape(Xtraint.shape[0], Xtraint.shape[1] * Xtraint.shape[2])
        ytrain = ytrain.reshape(Ytraint.shape[0], Ytraint.shape[1] * Ytraint.shape[2])

        xtest = xtest.reshape(Xtestt.shape[0], Xtestt.shape[1] * Xtestt.shape[2])
        ytest = ytest.reshape(Ytestt.shape[0], Ytestt.shape[1] * Ytestt.shape[2])

    # Check a baseline first
    start_time = time.time()
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)
    print(f"TIME train use XGBRegressor, {time.time() - start_time} , seconds.")

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
    #print_dq([traindl2st, testdl2st], ["train2st", "test2st"], -1.0)
    #vbtrain = print_nor_dq_filter0clip("trainnor", [traindl2st], ["train2st"], traindlrand, traindltrue)
    #vbtest = print_nor_dq_filter0clip("testnor", [testdl2st], ["test2st"], testdlrand, testdltrue)


    search_map_cv = {"wmse": WeightedLossCrossValidation, "mse++": DirectedLossCrossValidation, "idx": SearchbyInstanceCrossValid, "msehyp++": DirectedLossCrossValHyper, "quad": QuadLossCrossValidation}

    search_model = search_map_cv[params["search_method"]]
    model = search_model(prob, params, xtrain, ytrain, args.param_low, args.param_upp, args.param_def, nfold=params["cv_fold"], eta=params["xgb_lr"])

    booster, bltraindl, bltestdl = test_config_vec(params, prob, model.get_xgb_params(), model.get_def_loss_fn().get_obj_fn(), xtrain, ytrain, None, xtest, ytest, None, desc="defparams")

    print_dq([traindl2st, testdl2st, bltraindl, bltestdl], ["train2st", "test2st", "trainbl", "testbl"], -1.0)
    print_nor_dq_filter0clip("trainnor", [traindl2st, bltraindl], ["train2st", "trainbl"], traindlrand, traindltrue)
    print_nor_dq_filter0clip("testnor", [testdl2st, bltestdl], ["test2st", "testbl"], testdlrand, testdltrue)

    print_nor_dqagg("trainnor", [traindl2st, bltraindl], ["train2st", "trainbl"], traindlrand, traindltrue)
    print_nor_dqagg("testnor", [testdl2st, bltestdl], ["test2st", "testbl"], testdlrand, testdltrue)


    if params["test_nn2st"] != "none":
        from train_dense import nn2st_iter, perf_nn
        start_time = time.time()
        model = nn2st_iter(prob, xtrain, ytrain, None, None, params["nn_lr"], params["nn_iters"], params["batchsize"], params["n_layers"], params["int_size"], model_type=params["test_nn2st"], print_freq=(1+params["n_test_history"]))
        print(f"TIME train nn2st takes, {time.time() - start_time}, seconds")
        # Here just use the same data for tuning
        nntestdl = perf_nn(prob, model, xtest, ytest, None)
        nntraindl = perf_nn(prob, model, xtrain, ytrain, None)

        print_dq([nntraindl, nntestdl], ["NN2stTrainval", "NN2stTest"], -1.0)
        print_nor_dq_filter0clip("nn2stTrainNorDQ", [nntraindl], ["NN2stTrain"], traindlrand, traindltrue)
        print_nor_dq_filter0clip("nn2stTestNorDQ", [nntestdl], ["NN2stTestdl"], testdlrand, testdltrue)

        print_nor_dqagg("nn2stTrainNorDQ_", [nntraindl], ["NN2stTrain"], traindlrand, traindltrue)
        print_nor_dqagg("nn2stTestNorDQ_", [nntestdl], ["NN2stTestdl"], testdlrand, testdltrue)
        exit(0)

    if params["baseline"] == "lancer":
        from lancer_learner import test_lancer
        model, lctraindl, lctestdl = test_lancer(prob, xtrain, ytrain, None, xtest, ytest, None,
                                                lancer_in_dim=prob.num_feats, c_out_dim=prob.num_feats, n_iter=10, c_max_iter=5, c_nbatch=128,
                                                lancer_max_iter=5, lancer_nbatch=1024, c_epochs_init=30, c_lr_init=0.005,
                                                lancer_lr=0.001, c_lr=0.005, lancer_n_layers=2, lancer_layer_size=100, c_n_layers=0, c_layer_size=64,
                                                lancer_weight_decay=0.01, c_weight_decay=0.01, z_regul=0.0,
                                                lancer_out_activation="relu", c_hidden_activation="tanh", c_output_activation="relu", print_freq=(1+params["n_test_history"]),
                                                allowdiffzf=True if params["use_decouple"] else False)

        print_dq([lctraindl, lctestdl], ["LANCERtrain", "LANCERtest"], -1.0)
        print_nor_dq_filter0clip("LANCERTrainNorDQ", [lctraindl], ["LANCERTrain"], traindlrand, traindltrue)
        print_nor_dq_filter0clip("LANCERTestNorDQ", [lctestdl], ["LANCERTestdl"], testdlrand, testdltrue)

        print_nor_dqagg("LANCERTrainNorDQ_", [lctraindl], ["LANCERTrain"], traindlrand, traindltrue)
        print_nor_dqagg("LANCERTestNorDQ_", [lctestdl], ["LANCERTestdl"], testdlrand, testdltrue)
        exit(0)

    scenario = Scenario(model.configspace, n_trials=params["n_trials"])
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
            _, itertrain, itertest = test_config_vec(params, prob, model.get_xgb_params(), model.get_loss_fn(info.config).get_obj_fn(), xtrain, ytrain, None, xtest, ytest, None)
            print(f"iter{cnt}: cost is {cost} config is {model.get_vec(info.config)}")
            print_dq([itertrain, itertest], [f"iter{cnt}train", f"iter{cnt}test"], -1.0)
            print_nor_dq_filter0clip(f"iternordqtrain", [itertrain], [f"iter{cnt}train"], traindlrand, traindltrue)
            print_nor_dq_filter0clip(f"iternordqtest", [itertest], [f"iter{cnt}test"], testdlrand, testdltrue)
            print_nor_dqagg(f"iternordqtrain_", [itertrain], [f"iter{cnt}train"], traindlrand, traindltrue)
            print_nor_dqagg(f"iternordqtest_", [itertest], [f"iter{cnt}test"], testdlrand, testdltrue)
            # Check perf if stop now
            candidatesit = sorted(records, key=lambda x : x[0])
            select = len(candidatesit) - 1
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
            print_nor_dq_filter0clip(f"SFnordqtrain", [itertrain], [f"SF{cnt}train"], traindlrand, traindltrue)
            print_nor_dq_filter0clip(f"SFnordqtest", [itertest], [f"SF{cnt}test"], testdlrand, testdltrue)
            print_nor_dqagg(f"SFnordqtrain_", [itertrain], [f"SF{cnt}train"], traindlrand, traindltrue)
            print_nor_dqagg(f"SFnordqtest_", [itertest], [f"SF{cnt}test"], testdlrand, testdltrue)




    candidates = sorted(records, key=lambda x : x[0])
    select = len(candidates) - 1
    for i in range(1, len(candidates)):
        if candidates[i][0] != candidates[0][0]:
            select = i
            print(f"from idx 0 to {select} has the same cost randomly pick one")
            break
    idx = random.randint(0, select - 1)
    incumbent = candidates[idx][1]
    print(f"TIME Search takes {time.time() - start_time} seconds")

    params_vec = model.get_vec(incumbent)
    print(f"print {incumbent}")
    print(f"Seaerch Choose {params_vec}")

    start_time = time.time()
    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train(model.get_xgb_params(), dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())
    print(f"TIME Final train time {time.time() - start_time} seconds")

    print_booster_mse(f"mseFinaltrain", booster, xtrain, ytrain)
    print_booster_mse(f"mseFinaltest", booster, xtest, ytest)
    print("")

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain).flatten()

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest).flatten()

    _, bltrainfirst, bltestfirst = test_config_vec(params, prob, model.get_xgb_params(), model.get_loss_fn(records[0][1]).get_obj_fn(), xtrain, ytrain, None, xtest, ytest, None, desc="search1st") # Check the performance of the first iteration

    print_dq([trainsmac, testsmac, bltestdl, bltrainfirst, bltestfirst], ["trainsmac", "testsmac", "bldef", "bltrainfirst", "bltestfirst"], -1.0)
    print_nor_dq_filter0clip("Comparetrainnor", [traindl2st, trainsmac], ["traindl2st", "trainsmac"], traindlrand, traindltrue)
    print_nor_dq_filter0clip("Comparetestnor", [testdl2st, testsmac, bltestdl, bltestfirst], ["testdl2st", "testsmac", "bltestdl", "blfirst"], testdlrand, testdltrue)

    print_nor_dqagg("Aggparetrainnor", [traindl2st, trainsmac], ["traindl2st", "trainsmac"], traindlrand, traindltrue)
    print_nor_dqagg("Aggparetestnor", [testdl2st, testsmac, bltestdl, bltestfirst], ["testdl2st", "testsmac", "bltestdl", "blfirst"], testdlrand, testdltrue)


