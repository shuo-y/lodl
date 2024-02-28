import random
import argparse
import numpy as np
import torch
import xgboost as xgb

# From SMAC examples

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from train_xgb import search_weights_loss, search_quadratic_loss

# QuadLoss is based on SMAC examples https://automl.github.io/SMAC3/v2.0.2/examples/1_basics/2_svm_cv.html
class QuadLoss:
    def __init__(self, prob, params, xtrain, ytrain, xval, yval, valtruedl):
        self.Xy = xgb.DMatrix(xtrain, ytrain)
        self.xval = xval
        self.yval = yval
        self.params = params
        self.valtruedl = valtruedl
        self.prob = prob


    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        cs = ConfigurationSpace(seed=0)
        w1 = Float("w1", (1, 10000), default=1)
        w2 = Float("w2", (1, 10000), default=1)
        w3 = Float("w3", (1, 10000), default=1)
        cs.add_hyperparameters([w1, w2, w3])

    def train(self, config: Configuration, seed: int) -> float:
        config_dict = config.get_dictionary()
        weight_vec = np.array([[config["w1"], 0], [config["w2"], config["w3"]]])

        cusloss = search_quadratic_loss(yval.shape[0], yval.shape[1], weight_vec, 0.1)
        booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": 2},
                             dtrain = self.Xy, num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

        yvalpred = booster.inplace_predict(self.xval)
        valdl = self.prob.dec_loss(yvalpred, self.yval)

        cost = (valdl - self.valtruedl).mean()
        return cost




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
    parser.add_argument("--n-trials", type=int, default=200)

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


    from ExampleProb import ExampleProb

    prob = ExampleProb()

    X, Y = prob._generate_dataset_single_fun(0, 1, args.num_train + args.num_val + args.num_test, quad_fun)
    X = np.expand_dims(X, axis=1)

    indices = list(range(args.num_train + args.num_val + args.num_test))
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]].squeeze()

    xval = X[indices[args.num_train:(args.num_train + args.num_val)]]
    yval = Y[indices[args.num_train:(args.num_train + args.num_val)]].squeeze()

    xtest = X[indices[(args.num_train + args.num_val):]]
    ytest = Y[indices[(args.num_train + args.num_val):]].squeeze()


    # Check a baseline first
    reg = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    reg.fit(xtrain, ytrain)

    xtrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(xtrainpred, xtrain)


    yvalpred = reg.predict(xval)
    valdl2st = prob.dec_loss(yvalpred, yval)

    traindltrue = prob.dec_loss(ytrain, ytrain)
    valdltrue = prob.dec_loss(yval, yval)
    testdltrue = prob.dec_loss(ytest, ytest)


    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest)


    model = QuadLoss(prob, params, xtrain, ytrain, xval, yval, valdltrue)
    scenario = Scenario(model.configspace, n_trials=args.n_trials)
    HPOFacade(scenario, model.train, overwrite=True)
    incumbent = smac.optimize()


    weight_vec = np.array([[incumbent["w1"], 0], [incumbent["w2"], incumbent["w3"]]])

    cusloss = search_quadratic_loss(yval.shape[0], yval.shape[1], weight_vec, 0.1)
    booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": 2},
                             dtrain = self.Xy, num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

    res_str= [(f"2stageTrainDL,2stageTrainDLstderr,2stageValDL,2stageValDLstderr,2stageTestDL,2stageTestDLstderr,"
               f"smacTrainDL,smacTrainDLsstderr,smacValDL,smacValDLstderr,smacTestDL,smacTestDLstderr")]
    res_str.append((f"{}"))



