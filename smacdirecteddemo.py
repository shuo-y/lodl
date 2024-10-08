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
from smac.runhistory.dataclasses import TrialValue
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss
from PThenO import PThenO
from smacdirected import DirectedLoss, QuadSearch, DirectedLossMag, test_config

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

    def rand_loss(self, z_true: np.ndarray) -> np.ndarray:
        rand_dec = np.random.randint(2, size=len(z_true))
        rand_dec = rand_dec * 2 - 1
        obj = np.apply_along_axis(np.prod, 1, z_true) * rand_dec
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
        """
        for i in range(N):
            # cost without noise

            xi = (np.dot(B, z[i].reshape(d, 1)).T / np.sqrt(num_feats) + 3) ** deg + 1
            # rescale
            xi /= 3.5 ** deg
            # noise
            epislon = np.random.uniform(1 - noise_width, 1 + noise_width, num_feats)
            xi *= epislon
            x[i, :] = xi
        """
        return x, z

    def checky(self, y):
        yprod = np.apply_along_axis(np.prod, 1, y)
        print(f"{y[:, 0].mean()},{y[:, 1].mean()},{yprod.mean()}")
        if not (y[:, 0].mean() < 0 and y[:, 1].mean() < 0 and yprod.mean() < 0):
            print(f"Warning: sign of E[y0]E[y1] is the same as E[y0y1]")


    def get_decision(self, y_pred: np.ndarray, **kwargs):
        def opt(yi):
            # min y0 y1 c
            # c \in {-1, 1}
            if yi[0] * yi[1] >= 0:
                c = -1
            else:
                c = 1
            return c

        return np.apply_along_axis(opt, 1, y_pred)


    def get_modelio_shape(self):
        pass

    def get_objective(self, y_vec: np.ndarray, dec: np.ndarray, **kwargs):
        return np.apply_along_axis(np.prod, 1, y_vec) * dec

    def get_best_const(self, y):
        dl1 = self.get_objective(y, np.array([-1 for _ in range(len(y))])).mean()
        dl2 = self.get_objective(y, np.array([1 for _ in range(len(y))])).mean()
        return min(dl1, dl2)

    def get_output_activation(self):
        pass


def compute_stderror(vec: np.ndarray) -> float:
    popstd = vec.std()
    n = len(vec)
    return (popstd * np.sqrt(n / (n - 1.0))) / np.sqrt(n)

def sanity_check(vec: np.ndarray, msg: str) -> None:
    if (vec < 0).any():
        print(f"{msg}: check some negative value {vec}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "approx", "auto", "exact"])
    parser.add_argument("--search-estimators", type=int, default=100)
    parser.add_argument("--output", type=str, default="two_quad_example")
    parser.add_argument("--search-method", type=str, default="quad", choices=["mse++", "msemag++", "quad"])
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--num-test", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--param-low", type=float, default=0.0001)
    parser.add_argument("--param-upp", type=float, default=0.01)
    parser.add_argument("--param-def", type=float, default=0.001)
    parser.add_argument("--test-history", action="store_true", help="Check test dl of the history")

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

    ytrainpred = reg.predict(xtrain)
    traindl2st = prob.dec_loss(ytrainpred, ytrain)

    yvalpred = reg.predict(xval)
    valdl2st = prob.dec_loss(yvalpred, yval)

    traindltrue = prob.dec_loss(ytrain, ytrain)
    valdltrue = prob.dec_loss(yval, yval)
    testdltrue = prob.dec_loss(ytest, ytest)

    traindlconsttrue = prob.get_best_const(ytrain)
    valdlconsttrue = prob.get_best_const(yval)
    testdlconsttrue = prob.get_best_const(ytest)

    traindlrand = prob.rand_loss(ytrain)
    valdlrand = prob.rand_loss(yval)
    testdlrand = prob.rand_loss(ytest)

    ytestpred = reg.predict(xtest)
    testdl2st = prob.dec_loss(ytestpred, ytest)

    search_map = {"mse++": DirectedLoss, "quad": QuadSearch, "msemag++": DirectedLossMag}
    search_model = search_map[args.search_method]

    model = search_model(prob, params, xtrain, ytrain, xval, yval, valdltrue, None, args.param_low, args.param_upp, args.param_def)
    scenario = Scenario(model.configspace, n_trials=args.n_trials)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)  # We basically use one seed per config only

    smac = HPOFacade(scenario, model.train, intensifier=intensifier, overwrite=True)

    # We can ask SMAC which trials should be evaluated next
    history_val = []

    for cnt in range(args.n_trials):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost)

        smac.tell(info, value)

        if args.test_history:
            testdl, testvar = test_config(params, prob, model, xtrain, ytrain, xtest, ytest, testdltrue, info.config)
            history_val.append((cost, testdl, testvar))

    final = smac.ask()
    incumbent = final.config


    #incumbent = smac.optimize()


    weight_vec = model.get_vec(incumbent)
    print(f"SMAC choose {weight_vec}")

    cusloss = model.get_loss_fn(incumbent)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train({"tree_method": params["tree_method"], "num_target": 2},
                             dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())

    smacytrainpred = booster.inplace_predict(xtrain)
    trainsmac = prob.dec_loss(smacytrainpred, ytrain)

    smacyvalpred = booster.inplace_predict(xval)
    valsmac = prob.dec_loss(smacyvalpred, yval)

    smacytestpred = booster.inplace_predict(xtest)
    testsmac = prob.dec_loss(smacytestpred, ytest)

    sanity_check(traindl2st - traindltrue, "train2st")
    sanity_check(valdl2st - valdltrue, "val2st")
    sanity_check(testdl2st - testdltrue, "test2st")
    sanity_check(trainsmac - traindltrue, "trainsmac")
    sanity_check(valsmac - valdltrue, "valsmac")
    sanity_check(testsmac - testdltrue, "testsmac")
    sanity_check(traindlrand - traindltrue, "trainrand")
    sanity_check(valdlrand - valdltrue, "valrand")
    sanity_check(testdlrand - testdltrue, "testrand")


    res_str= [(f"title,2stageTrainDL,2stageTrainDLstderr,2stageValDL,2stageValDLstderr,2stageTestDL,2stageTestDLstderr,"
               f"smacTrainDL,smacTrainDLsstderr,smacValDL,smacValDLstderr,smacTestDL,smacTestDLstderr,"
               f"randTrainDL,randTrainDLsstderr,randValDL,randValDLstderr,randTestDL,randTestDLstderr,"
               f"constTrainDL,constTrainDLsstderr,constValDL,constValDLstderr,constTestDL,constTestDLstderr")]
    res_str.append((f"res, {(traindl2st - traindltrue).mean()}, {compute_stderror(traindl2st - traindltrue)}, "
                    f"{(valdl2st - valdltrue).mean()}, {compute_stderror(valdl2st - valdltrue)}, "
                    f"{(testdl2st - testdltrue).mean()}, {compute_stderror(testdl2st - testdltrue)}, "
                    f"{(trainsmac - traindltrue).mean()}, {compute_stderror(trainsmac - traindltrue)}, "
                    f"{(valsmac - valdltrue).mean()}, {compute_stderror(valsmac - valdltrue)}, "
                    f"{(testsmac - testdltrue).mean()}, {compute_stderror(testsmac - testdltrue)},"
                    f"{(traindlrand - traindltrue).mean()}, {compute_stderror(traindlrand - traindltrue)}, "
                    f"{(valdlrand - valdltrue).mean()}, {compute_stderror(valdlrand - valdltrue)}, "
                    f"{(testdlrand - testdltrue).mean()}, {compute_stderror(testdlrand - testdltrue)}, "
                    f"{(traindlconsttrue - traindltrue).mean()}, {compute_stderror(traindlconsttrue - traindltrue)}, "
                    f"{(valdlconsttrue - valdltrue).mean()}, {compute_stderror(valdlconsttrue - valdltrue)}, "
                    f"{(testdlconsttrue - testdltrue).mean()}, {compute_stderror(testdlconsttrue - testdltrue)}"))


    for row in res_str:
        print(row)

        #TODO how Lower L map to y0y1



    if args.test_history:
        for valdl, testdl, testvar in history_val:
            print(f"{valdl},{testdl},{testvar}")




