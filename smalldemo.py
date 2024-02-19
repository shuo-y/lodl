import random
import argparse
import numpy as np
import torch
import xgboost as xgb
from train_xgb import search_weights_loss, search_quadratic_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--search_estimators", type=int, default=100)
    parser.add_argument("--output", type=str, default="two_quad_example")
    parser.add_argument("--loss", type=str, default="quad", choices=["mse", "quad"])
    parser.add_argument("--num-train", type=int, default=20)
    parser.add_argument("--num-val", type=int, default=20)
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--steiner-degree", type=int, default=0)

    args = parser.parse_args()

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


    from ExampleProb import ExampleProb

    prob = ExampleProb()

    X, Y = prob._generate_dataset_single_fun(0, 1, 80, quad_fun)
    X = np.expand_dims(X, axis=1)

    indices = list(range(args.num_train + args.num_val + args.num_test))
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]].squeeze()

    xval = X[indices[args.num_train:(args.num_train + args.num_val)]]
    yval = Y[indices[args.num_train:(args.num_train + args.num_val)]].squeeze()

    xtest = X[indices[(args.num_train + args.num_val):]]
    ytest = Y[indices[(args.num_train + args.num_val):]].squeeze()

    data = []
    for i in range(args.num_train):
        data.append(f"train,{xtrain[i][0]}, {ytrain[i][0]}, {ytrain[i][1]}")

    for i in range(args.num_val):
        data.append(f"val,{xval[i][0]}, {yval[i][0]}, {yval[i][1]}")

    for i in range(args.num_test):
        data.append(f"test,{xtest[i][0]}, {ytest[i][0]}, {ytest[i][1]}")

    # Check a baseline first
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xval, yval)
    baselinedl = reg.predict(xval)
    valdl = prob.dec_loss(baselinedl, yval, steiner=args.steiner_degree).mean()

    ytestpred = reg.predict(xtest)
    testdl = prob.dec_loss(ytestpred, ytest, steiner=args.steiner_degree).mean()
    print(f"Baseline on val: {valdl} test: {testdl}")
    tables = []

    if args.loss == "mse":
        for w1 in np.logspace(1.0, 10.0, num=50, base=10):
            for w2 in np.logspace(1.0, 10.0, num=50, base=10):
                weight_vec = np.array([w1, w2])
                cusloss = search_weights_loss(yval.shape[0], yval.shape[1], weight_vec)
                Xy = xgb.DMatrix(xval, yval)

                booster = xgb.train({"tree_method": params["tree_method"],
                                    "num_target": 2},
                                    dtrain = Xy,
                                    num_boost_round = params["search_estimators"],
                                    obj = cusloss.get_obj_fn())

                ypred = booster.inplace_predict(xtrain)
                traindl = prob.dec_loss(ypred, ytrain, steiner=args.steiner_degree).mean()

                yvalpred = booster.inplace_predict(xval)
                valdl = prob.dec_loss(yvalpred, yval, steiner=args.steiner_degree).mean()

                ytestpred = booster.inplace_predict(xtest)
                testdl = prob.dec_loss(ytestpred, ytest, steiner=args.steiner_degree).mean()
                #print(f"w1 {w1} w2 {w2} train dl{traindl} test dl{testdl}")
                csvstring = "%.12f,%.12f,%.12f,%.12f,%.12f" % (w1, w2, traindl, valdl, testdl)
                tables.append(csvstring)
    else:
        for w1 in np.logspace(1.0, 10.0, num=args.grid_size, base=10):
            for w2 in np.logspace(1.0, 10.0, num=args.grid_size, base=10):
                for w3 in np.logspace(1.0, 10.0, num=args.grid_size, base=10):
                    weight_vec = np.array([[w1, 0], [w2, w3]])
                    cusloss = search_quadratic_loss(yval.shape[0], yval.shape[1], weight_vec, 0.1)
                    #print(cusloss.basis @ cusloss.basis.T)
                    #import pdb
                    #pdb.set_trace()
                    Xy = xgb.DMatrix(xval, yval)

                    booster = xgb.train({"tree_method": params["tree_method"],
                                        "num_target": 2},
                                        dtrain = Xy,
                                        num_boost_round = params["search_estimators"],
                                        obj = cusloss.get_obj_fn())

                    ypred = booster.inplace_predict(xtrain)
                    traindl = prob.dec_loss(ypred, ytrain, steiner=args.steiner_degree)
                    traindltrue = prob.dec_loss(ytrain, ytrain, steiner=args.steiner_degree)

                    yvalpred = booster.inplace_predict(xval)
                    valdl = prob.dec_loss(yvalpred, yval, steiner=args.steiner_degree)
                    valdltrue = prob.dec_loss(yval, yval, steiner=args.steiner_degree)

                    ytestpred = booster.inplace_predict(xtest)
                    testdl = prob.dec_loss(ytestpred, ytest, steiner=args.steiner_degree)
                    testdltrue = prob.dec_loss(ytest, ytest, steiner=args.steiner_degree)
                    #print(f"w1 {w1} w2 {w2} train dl{traindl} test dl{testdl}")
                    csvstring = f"""{w1:.12f},{w2:.12f},{w3:.12f},
                    {traindl.mean():.12f}, {traindltrue.mean():.12f}, {(traindl - traindltrue).min():.12f},
                    {valdl.mean():.12f}, {valdltrue.mean():.12f}, {(valdl - valdltrue).min():.12f},
                    {testdl.mean():.12f}, {testdltrue.mean():.12f}, {(testdl - testdltrue).min():.12f}
                    """

                    tables.append(csvstring)

    rep_str = f"{args.output}.{args.num_train}.{args.num_val}.{args.num_test}.par{args.steiner_degree}"
    datafile = open(f"{rep_str}.data.csv", "w")
    for row in data:
        print(row, file=datafile)

    tablefile = open(f"{rep_str}.table.csv", "w")
    for row in tables:
        print(row, file=tablefile)
        # infinite sample
