import random
import argparse
import numpy as np
import torch
import xgboost as xgb
from train_xgb import search_weights_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree_method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--search_estimators", type=int, default=100)
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


    from ExampleProb import ExampleProb

    prob = ExampleProb()

    X, Y = prob._generate_dataset_single_fun(-1, 1, 150, rand_fun)
    X = np.expand_dims(X, axis=1)
    xtrain = X[:50]
    ytrain = Y[:50]

    xval = X[50:100]
    yval = Y[50:100]

    xtest = X[100:]
    ytest = Y[100:]

    tables = []

    for w1 in np.logspace(1.0, 10.0, num=50, base=10):
        for w2 in np.logspace(1.0, 10.0, num=50, base=10):
            weight_vec = np.array([w1, w2])
            cusloss = search_weights_loss(ytrain.shape[0], ytrain.shape[1], weight_vec)
            Xy = xgb.DMatrix(xtrain, ytrain)

            booster = xgb.train({"tree_method": params["tree_method"],
                                 "num_target": 2},
                                dtrain = Xy,
                                num_boost_round = params["search_estimators"],
                                obj = cusloss.get_obj_fn())

            ypred = booster.inplace_predict(xtrain)
            traindl = prob.dec_loss(ypred, ytrain).mean()

            ytestpred = booster.inplace_predict(xtest)
            testdl = prob.dec_loss(ytestpred, ytest).mean()
            print(f"w1 {w1} w2 {w2} train dl{traindl} test dl{testdl}")
            csvstring = "%.12f,%.12f,%.12f,%.12f" % (w1, w2, traindl, testdl)
            tables.append(csvstring)

    for row in tables:
        print(row)
