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
from PThenO import PThenO

# 2-dimensional Rosenbrock function https://automl.github.io/SMAC3/v2.0.2/examples/1_basics/3_ask_and_tell.html
class ProdObj(PThenO):
    def __init__(self):
        super(ProdObj, self).__init__()
        pass

    def dec_loss(self, z_pred: np.ndarray, z_true: np.ndarray, verbose=False, **kwargs) -> np.ndarray:

        def opt(yi):
            # min y0 y1 c
            # c \in {-1, 1}
            if yi[0] * yi[1] >= 0:
                c = -1
            else:
                c = 1
            return c

        dec = np.apply_along_axis(opt, 1, z_pred)
        obj = np.apply_along_axis(np.prod, 1, z_true) * dec
        return obj



    def generate_dataset(self, N, deg, noise_width,
                         num_feats, d=2, mean=np.array([-0.9, -0.9]),
                         cov=np.array([[1, -1], [-1, 5]])):
        """
        From the LANCER code
        Generate synthetic dataset for the DFL shortest path problem
        :param N: number of points
        :param deg: degree of polynomial to enforce nonlinearity
        :param noise_width: add eps noise to the cost vector
        :return: dataset of features x and the ground truth cost vector of edges c
        """
        # random matrix parameter B
        B = np.random.binomial(1, 0.5, (num_feats, d))
        # feature vectors
        x = np.random.normal(0, 1, (N, num_feats))
        # cost vectors
        z = np.random.multivariate_normal(mean, cov, N)
        for i in range(N):
            # cost without noise

            xi = (np.dot(B, z[i].reshape(d, 1)).T / np.sqrt(num_feats) + 3) ** deg + 1
            # rescale
            xi /= 3.5 ** deg
            # noise
            epislon = np.random.uniform(1 - noise_width, 1 + noise_width, num_feats)
            xi *= epislon
            x[i, :] = xi
        return x, z

    def checky(self, y):
        yprod = np.apply_along_axis(np.prod, 1, y)
        print(f"{y[:, 0].mean()},{y[:, 1].mean()},{yprod.mean()}")
        if not (y[:, 0].mean() < 0 and y[:, 1].mean() < 0 and yprod.mean() < 0):
            print(f"Warning: sign of E[y0]E[y1] is the same as E[y0y1]")


    def get_decision(self):
        pass

    def get_modelio_shape(self):
        pass

    def get_objective(self):
        pass

    def get_output_activation(self):
        pass


def compute_stderror(vec: np.ndarray) -> float:
    popstd = vec.std()
    n = len(vec)
    return (popstd * np.sqrt(n / (n - 1.0))) / np.sqrt(n)

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
        return cs

    def train(self, config: Configuration, seed: int) -> float:
        weight_vec = np.array([[config["w1"], 0], [config["w2"], config["w3"]]])

        cusloss = search_quadratic_loss(ytrain.shape[0], ytrain.shape[1], weight_vec, 0.1)
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
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--num-test", type=int, default=2000)
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


    N = args.num_train + args.num_val + args.num_test
    prob = ProdObj()
    X, Y = prob.generate_dataset(N, 6, 0.5, 10)

    prob.checky(Y)

    indices = list(range(args.num_train + args.num_val + args.num_test))
    np.random.shuffle(indices)

    xtrain = X[indices[:args.num_train]]
    ytrain = Y[indices[:args.num_train]].squeeze()

    xval = X[indices[args.num_train:(args.num_train + args.num_val)]]
    yval = Y[indices[args.num_train:(args.num_train + args.num_val)]].squeeze()

    xtest = X[indices[(args.num_train + args.num_val):]]
    ytest = Y[indices[(args.num_train + args.num_val):]].squeeze()

    print("check ytrain")
    prob.checky(ytrain)
    print("check yval")
    prob.checky(yval)
    print("check ytest")
    prob.checky(ytest)


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
    smac = HPOFacade(scenario, model.train, overwrite=True)
    incumbent = smac.optimize()


    weight_vec = np.array([[incumbent["w1"], 0], [incumbent["w2"], incumbent["w3"]]])

    cusloss = search_quadratic_loss(ytrain.shape[0], ytrain.shape[1], weight_vec, 0.1)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train({"tree_method": params["tree_method"], "num_target": 2},
                             dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain)

    smacyvalpred = booster.inplace_predict(xval)
    valsmac = prob.dec_loss(smacyvalpred, yval)

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest)

    res_str= [(f"2stageTrainDL,2stageTrainDLstderr,2stageValDL,2stageValDLstderr,2stageTestDL,2stageTestDLstderr,"
               f"smacTrainDL,smacTrainDLsstderr,smacValDL,smacValDLstderr,smacTestDL,smacTestDLstderr")]
    res_str.append((f"{(traindl2st - traindltrue).mean()}, {compute_stderror(traindl2st - traindltrue)}, "
                    f"{(valdl2st - valdltrue).mean()}, {compute_stderror(valdl2st - valdltrue)}, "
                    f"{(testdl2st - testdltrue).mean()}, {compute_stderror(testdl2st - testdltrue)},"
                    f"{(trainsmac - traindltrue).mean()}, {compute_stderror(trainsmac - traindltrue)}, "
                    f"{(valsmac - valdltrue).mean()}, {compute_stderror(valsmac - valdltrue)}, "
                    f"{(testsmac - testdltrue).mean()}, {compute_stderror(testsmac - testdltrue)}"))

    for row in res_str:
        print(row)


