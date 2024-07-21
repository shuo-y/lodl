import random
import argparse
import numpy as np
import torch
import xgboost as xgb
import lightgbm as lgb
import copy
import time

# From SMAC examples

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.dataclasses import TrialValue
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss, search_quadratic_loss, search_full_weights
from PThenO import PThenO
from sklearn.multioutput import MultiOutputRegressor

# QuadLoss is based on SMAC examples https://automl.github.io/SMAC3/v2.0.2/examples/1_basics/2_svm_cv.html
class DirectedLoss:
    def __init__(self, prob, params, xtrain, ytrain, xval, yval, param_low, param_upp, param_def, auxdata=None, reg2st=None, use_vec=False, initvec=None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xval = xval
        self.yval = yval
        self.params = params
        self.prob = prob
        self.aux_data = auxdata
        self.param_low = param_low
        self.param_upp = param_upp
        self.param_def = param_def
        self.reg2stmodel = reg2st
        self.use_vec = use_vec
        self.initvec = initvec
        # 0.0001, 0.01, 0.001


    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        if self.use_vec == True:
            configs = [Float(f"w{i}", (self.param_low, self.param_upp), default=self.initvec[i]) for i in range(len(self.initvec))]
        else:
            configs = [Float(f"w{i}", (self.param_low, self.param_upp), default=self.param_def) for i in range(2 * self.yval.shape[1])]
        cs.add_hyperparameters(configs)
        return cs

    def train(self, configs: Configuration, seed: int) -> float:
        configarray = [configs[f"w{i}"] for i in range(2 * self.yval.shape[1])]
        weight_vec = np.array(configarray)
        Xy = xgb.DMatrix(self.xtrain, self.ytrain)

        cusloss = search_weights_directed_loss(weight_vec)
        if self.reg2stmodel == None:
            booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]},
                                dtrain = Xy, num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())
        else:
            xgbmodel = self.reg2stmodel.get_booster().copy()
            booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]},
                                dtrain = Xy, num_boost_round = self.params["search_estimators"], xgb_model = xgbmodel, obj = cusloss.get_obj_fn())

        yvalpred = booster.inplace_predict(self.xval)
        valdl = self.prob.dec_loss(yvalpred, self.yval, aux_data=self.aux_data)

        return valdl.mean()

    def train_lgb(self, configs: Configuration, seed: int) -> float:
        configarray = [configs[f"w{i}"] for i in range(2 * self.yval.shape[1])]
        weight_vec = np.array(configarray)
        xy_lgb = lgb.Dataset(self.xtrain, self.ytrain)

        cusloss = search_weights_directed_loss(weight_vec)
        lgb_model = lgb.train({"boosting_type": "gbdt", "objective": cusloss.get_obj_fn()}, xy_lgb, num_boost_round=self.params["search_estimators"])
        yvalpred = lgb_model.predict(self.xval)
        valdl = self.prob.dec_loss(yvalpred, self.yval, aux_data=self.aux_data)

        return valdl.mean()

    def get_vec(self, incumbent) -> np.ndarray:
        # Get the weight vector from incumbent
        arr = [incumbent[f"w{i}"] for i in range(2 * self.yval.shape[1])]
        return np.array(arr)

    def get_loss_fn(self, incumbent):
        arr = [incumbent[f"w{i}"] for i in range(2 * self.yval.shape[1])]
        weight_vec = np.array(arr)
        cusloss = search_weights_directed_loss(weight_vec)
        return cusloss

    def get_def_loss_fn(self):
        arr = [self.param_def for i in range(2 * self.yval.shape[1])]
        weight_vec = np.array(arr)
        cusloss = search_weights_directed_loss(weight_vec)
        return cusloss

    def get_xgb_params(self):
        return {"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]}

class SearchbyInstance:
    def __init__(self, prob, params, xtrain, ytrain, xval, yval, param_low, param_upp, param_def, auxdata=None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xval = xval
        self.yval = yval
        self.params = params
        self.prob = prob
        self.aux_data = auxdata
        self.param_low = param_low
        self.param_upp = param_upp
        self.param_def = param_def
        # 0.0001, 0.01, 0.001

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        configs = [Float(f"w{i}", (self.param_low, self.param_upp), default=self.param_def) for i in range(self.ytrain.shape[0])]
        cs.add_hyperparameters(configs)
        return cs

    def train(self, configs: Configuration, seed: int) -> float:
        configarray = [configs[f"w{i}"] for i in range(self.ytrain.shape[0])]
        weight_vec = np.array(configarray)
        weight_mat = np.tile(weight_vec, (self.ytrain.shape[1], 1)).T
        Xy = xgb.DMatrix(self.xtrain, self.ytrain)
        cusloss = search_full_weights(weight_mat)
        booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]},
                                dtrain = Xy, num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

        yvalpred = booster.inplace_predict(self.xval)
        valdl = self.prob.dec_loss(yvalpred, self.yval, aux_data=self.aux_data)

        return valdl.mean()


    def get_vec(self, incumbent) -> np.ndarray:
        # Get the weight vector from incumbent
        arr = [incumbent[f"w{i}"] for i in range(self.ytrain.shape[0])]
        return np.array(arr)

    def get_loss_fn(self, incumbent):
        configarray = [incumbent[f"w{i}"] for i in range(self.ytrain.shape[0])]
        weight_vec = np.array(configarray)
        weight_mat = np.tile(weight_vec, (self.ytrain.shape[1], 1)).T
        cusloss = search_full_weights(weight_mat)
        return cusloss

    def get_xgb_params(self):
        return {"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]}


class SearchbyInstanceCrossValid:
    def __init__(self, prob, params, X, y, param_low, param_upp, param_def, auxdata=None, nfold=5):
        self.params = params
        self.prob = prob
        self.aux_data = auxdata
        self.param_low = param_low
        self.param_upp = param_upp
        self.param_def = param_def
        self.Xys = []
        self.auxdatas = []
        self.valdatas = []
        N = len(X)
        cnt = N // nfold
        if N % nfold != 0:
            print("Warning: number of trainning items is not a multiple of the number of fold")
        self.nfold = nfold
        self.ydim = y.shape[1]
        self.nitems = N
        self.X = X
        self.y = y

        for i in range(self.nfold):
            testind = [idx for idx in range(i * cnt, (i + 1) * cnt)]
            otherind = [idx for idx in range(self.nitems) if idx < i * cnt or idx >= (i + 1) * cnt]
            self.Xys.append(xgb.DMatrix(X[otherind], y[otherind]))
            if auxdata is not None:
                self.auxdatas.append(auxdata[testind])
            self.valdatas.append((X[testind], y[testind]))

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        configs = [Float(f"w{i}", (self.param_low, self.param_upp), default=self.param_def) for i in range(self.nitems)]
        cs.add_hyperparameters(configs)
        return cs

    def train(self, configs: Configuration, seed: int) -> float:
        configarray = [configs[f"w{i}"] for i in range(self.nitems)]
        weight_vec = np.array(configarray)
        weight_mat = np.tile(weight_vec, (self.ydim, 1)).T

        costs = []
        cnt = self.nitems // self.nfold
        for i in range(self.nfold):
            otherind = [idx for idx in range(self.nitems) if idx < i * cnt or idx >= (i + 1) * cnt]
            cusloss = search_full_weights(weight_mat[otherind])
            booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.ydim},
                                    dtrain = self.Xys[i], num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

            yvalpred = booster.inplace_predict(self.valdatas[i][0])
            if len(self.auxdatas) > 0:
                valdl = self.prob.dec_loss(yvalpred, self.valdatas[i][1], aux_data=self.auxdatas[i])
            else:
                valdl = self.prob.dec_loss(yvalpred, self.valdatas[i][1])
            costs.append(valdl.mean())

        return np.mean(costs)

    def get_def_loss_fn(self):
        weight_vec = np.array([self.param_def for _ in range(self.nitems)])
        weight_mat = np.tile(weight_vec, (self.ydim, 1)).T
        cusloss = search_full_weights(weight_mat)
        return cusloss


    def get_vec(self, incumbent) -> np.ndarray:
        # Get the weight vector from incumbent
        arr = [incumbent[f"w{i}"] for i in range(self.nitems)]
        return np.array(arr)

    def get_loss_fn(self, incumbent):
        configarray = [incumbent[f"w{i}"] for i in range(self.nitems)]
        weight_vec = np.array(configarray)
        weight_mat = np.tile(weight_vec, (self.ydim, 1)).T
        cusloss = search_full_weights(weight_mat)
        return cusloss

    def get_xgb_params(self):
        return {"tree_method": self.params["tree_method"], "num_target": self.ydim}

class DirectedLossCrossValidation:
    def __init__(self, prob, params, X, Y, param_low, param_upp, param_def, auxdata=None, nfold=2, reg2st=None, use_vec=False, initvec=None):
        # directed loss with cross validation
        # just do not use xtrain ytrain and valtruedl
        self.params = params
        self.prob = prob
        self.param_low = param_low
        self.param_upp = param_upp
        self.param_def = param_def
        self.reg2stmodel = reg2st
        self.use_vec = use_vec
        self.initvec = initvec
        # 0.0001, 0.01, 0.001
        self.Xys = []
        self.auxdatas = []
        self.valdatas = []
        N = len(X)
        cnt = N // nfold
        self.nfold = nfold
        self.ydim = Y.shape[1]

        for i in range(self.nfold):
            testind = [idx for idx in range(i * cnt, (i + 1) * cnt)]
            otherind = [idx for idx in range(N) if idx < i * cnt or idx >= (i + 1) * cnt]
            self.Xys.append(xgb.DMatrix(X[otherind], Y[otherind]))
            if auxdata is not None:
                self.auxdatas.append(auxdata[testind])
            self.valdatas.append((X[testind], Y[testind]))


    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        if self.use_vec == True:
            configs = [Float(f"w{i}", (self.param_low, self.param_upp), default=self.initvec[i]) for i in range(len(self.initvec))]
        else:
            configs = [Float(f"w{i}", (self.param_low, self.param_upp), default=self.param_def) for i in range(2 * self.ydim)]
        cs.add_hyperparameters(configs)
        return cs

    def train(self, configs: Configuration, seed: int) -> float:
        configarray = [configs[f"w{i}"] for i in range(2 * self.ydim)]
        weight_vec = np.array(configarray)

        costs = []
        for i in range(self.nfold):
            cusloss = search_weights_directed_loss(weight_vec)
            booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.ydim},
                                    dtrain = self.Xys[i], num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

            yvalpred = booster.inplace_predict(self.valdatas[i][0])
            if len(self.auxdatas) > 0:
                valdl = self.prob.dec_loss(yvalpred, self.valdatas[i][1], aux_data=self.auxdatas[i])
            else:
                valdl = self.prob.dec_loss(yvalpred, self.valdatas[i][1])
            costs.append(valdl.mean())

        return np.mean(costs)

    def get_vec(self, incumbent) -> np.ndarray:
        # Get the weight vector from incumbent
        arr = [incumbent[f"w{i}"] for i in range(2 * self.ydim)]
        return np.array(arr)

    def get_loss_fn(self, incumbent):
        arr = [incumbent[f"w{i}"] for i in range(2 * self.ydim)]
        weight_vec = np.array(arr)
        cusloss = search_weights_directed_loss(weight_vec)
        return cusloss

    def get_xgb_params(self):
        return {"tree_method": self.params["tree_method"], "num_target": self.ydim}


# QuadLoss is based on SMAC examples https://automl.github.io/SMAC3/v2.0.2/examples/1_basics/2_svm_cv.html
class DirectedLossMag:
    def __init__(self, prob, params, xtrain, ytrain, xval, yval, valtruedl, auxdata, param_low, param_upp, param_def):
        self.Xy = xgb.DMatrix(xtrain, ytrain)
        self.xval = xval
        self.yval = yval
        self.params = params
        self.valtruedl = valtruedl
        self.prob = prob
        self.aux_data = auxdata
        self.param_low = param_low
        self.param_upp = param_upp
        self.param_def = param_def
        # 0.0001, 0.01, 0.001


    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        configs = [Float(f"w{i}", (self.param_low, self.param_upp), default=self.param_def) for i in range(2 * self.yval.shape[1])]
        configs.append(Float("magtitude", (0.001, 1000.0), default=1, log=True))
        cs.add_hyperparameters(configs)
        return cs

    def train(self, configs: Configuration, seed: int) -> float:
        configarray = [configs[f"w{i}"] for i in range(2 * self.yval.shape[1])]
        weight_vec = np.array(configarray)
        magtitude = configs["magtitude"]
        weight_vec = magtitude * weight_vec

        cusloss = search_weights_directed_loss(weight_vec)
        booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]},
                             dtrain = self.Xy, num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

        yvalpred = booster.inplace_predict(self.xval)
        valdl = self.prob.dec_loss(yvalpred, self.yval, aux_data=self.aux_data)

        return valdl.mean()

    def get_vec(self, incumbent) -> np.ndarray:
        # Get the weight vector from incumbent
        arr = [incumbent[f"w{i}"] for i in range(2 * self.yval.shape[1])]
        arr.append(incumbent["magtitude"])
        return np.array(arr)

    def get_loss_fn(self, incumbent):
        arr = [incumbent[f"w{i}"] for i in range(2 * self.yval.shape[1])]
        magtitude = incumbent["magtitude"]
        weight_vec = magtitude * np.array(arr)
        cusloss = search_weights_directed_loss(weight_vec)
        return cusloss

    def get_xgb_params(self):
        return {"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]}

class QuadSearch:
    def __init__(self, prob, params, xtrain, ytrain, xval, yval, valtruedl, auxdata, param_low, param_upp, param_def):
        self.Xy = xgb.DMatrix(xtrain, ytrain)
        self.xval = xval
        self.yval = yval
        self.params = params
        self.valtruedl = valtruedl
        self.prob = prob
        self.aux_data = auxdata
        self.total_params_n = ((1 + self.yval.shape[1]) * self.yval.shape[1]) // 2
        self.param_low = param_low
        self.param_upp = param_upp
        self.param_def = param_def
        # 0.0001, 1, 0.1


    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        configs = [Float(f"w{i}", (self.param_low, self.param_upp), default=self.param_def) for i in range(self.total_params_n)]
        cs.add_hyperparameters(configs)
        return cs

    def train(self, configs: Configuration, seed: int) -> float:
        configarray = [configs[f"w{i}"] for i in range(self.total_params_n)]

        indx = 0
        base_vec = np.zeros((self.yval.shape[1], self.yval.shape[1]))
        for i in range(self.yval.shape[1]):
            for j in range(i + 1):
                base_vec[i, j] = configarray[indx]
                indx += 1

        cusloss = search_quadratic_loss(base_vec, 0.01)
        booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1], "eta": 0.03},
                             dtrain = self.Xy, num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

        yvalpred = booster.inplace_predict(self.xval)
        valdl = self.prob.dec_loss(yvalpred, self.yval, aux_data=self.aux_data)

        return valdl.mean()

    def get_vec(self, incumbent) -> np.ndarray:
        # Get the weight vector from incumbent
        arr = [incumbent[f"w{i}"] for i in range(self.total_params_n)]
        indx = 0
        base_vec = np.zeros((self.yval.shape[1], self.yval.shape[1]))
        for i in range(self.yval.shape[1]):
            for j in range(i + 1):
                base_vec[i, j] = arr[indx]
                indx += 1
        return base_vec

    def get_loss_fn(self, incumbent):
        base_vec = self.get_vec(incumbent)
        cusloss = search_quadratic_loss(base_vec, 0.01)

        return cusloss

    def get_xgb_params(self):
        return {"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1], "eta": 0.03}


class XGBHyperSearch:
    def __init__(self, prob, X, Y, auxdata=None, nfold=5):
        self.prob = prob
        self.Xys = []
        self.auxdatas = []
        self.valdatas = []
        N = len(X)
        cnt = N // nfold
        self.nfold = nfold
        self.ydim = Y.shape[1]

        for i in range(self.nfold):
            testind = [idx for idx in range(i * cnt, (i + 1) * cnt)]
            otherind = [idx for idx in range(N) if idx < i * cnt or idx >= (i + 1) * cnt]
            self.Xys.append(xgb.DMatrix(X[otherind], Y[otherind]))
            if auxdata is not None:
                self.auxdatas.append(auxdata[testind])
            self.valdatas.append((X[testind], Y[testind]))


    @property
    def configspace(self) -> ConfigurationSpace:
        # Check the parameters from https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
        cs = ConfigurationSpace()
        eta = Float("eta", (0.01, 10.0), default=0.03)
        gamma = Float("gamma", (0.0, 100.0), default=0.0)
        max_depth = Integer("max_depth", (0, 100), default=6)
        min_child_weight = Float("min_child_weight", (0.0, 100.0), default=1.0)
        max_delta_step = Float("max_delta_step", (0.0, 100.0), default=0.0)
        subsample = Float("subsample", (0.00001, 1.0), default=1.0)
        #sampling_method = Categorical("sampling_method", ["uniform", "gradient_based"], default="uniform")
        colsample_bytree = Float("colsample_bytree", (0.00001, 1.0), default=1.0)
        colsample_bylevel = Float("colsample_bylevel", (0.00001, 1.0), default=1.0)
        colsample_bynode = Float("colsample_bynode", (0.00001, 1.0), default=1.0)
        reg_lambda = Float("lambda", (0, 100.0), default=1.0)
        alpha = Float("alpha", (0, 100.0), default=0.0)
        tree_method = Categorical("tree_method", ["auto", "exact", "approx", "hist"], default="auto")
        num_boost_round = Integer("num_boost_round", (1, 500), default=50)
        cs.add_hyperparameters([eta, gamma, max_depth, min_child_weight, max_delta_step, subsample,
                                colsample_bytree, colsample_bylevel, colsample_bynode, reg_lambda, alpha, tree_method, num_boost_round])
        return cs

    def train(self, config: Configuration, seed: int) -> float:
        params_dict = config.get_dictionary()
        num_boost_round = params_dict["num_boost_round"]
        del params_dict["num_boost_round"]

        costs = []
        for i in range(self.nfold):
            booster = xgb.train(params_dict, dtrain = self.Xys[i], num_boost_round = num_boost_round)
            yvalpred = booster.inplace_predict(self.valdatas[i][0])
            if len(self.auxdatas) > 0:
                valdl = self.prob.dec_loss(yvalpred, self.valdatas[i][1], aux_data=self.auxdatas[i])
            else:
                valdl = self.prob.dec_loss(yvalpred, self.valdatas[i][1])
            costs.append(valdl.mean())

        return np.mean(costs)


class XGBHyperSearchwDefault:
    def __init__(self, prob, X, Y, param_low, param_upp, param_def, auxdata=None, nfold=5):
        self.prob = prob
        self.Xys = []
        self.auxdatas = []
        self.valdatas = []
        N = len(X)
        cnt = N // nfold
        self.nfold = nfold
        self.ydim = Y.shape[1]

        weight_vec = np.array([param_def for _  in range(Y.shape[1])])
        self.def_fn = search_weights_loss(weight_vec)
        for i in range(self.nfold):
            testind = [idx for idx in range(i * cnt, (i + 1) * cnt)]
            otherind = [idx for idx in range(N) if idx < i * cnt or idx >= (i + 1) * cnt]
            self.Xys.append(xgb.DMatrix(X[otherind], Y[otherind]))
            if auxdata is not None:
                self.auxdatas.append(auxdata[testind])
            self.valdatas.append((X[testind], Y[testind]))


    @property
    def configspace(self) -> ConfigurationSpace:
        # Check the parameters from https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
        cs = ConfigurationSpace()
        eta = Float("eta", (0.01, 10.0), default=0.03)
        gamma = Float("gamma", (0.0, 100.0), default=0.0)
        max_depth = Integer("max_depth", (0, 100), default=6)
        min_child_weight = Float("min_child_weight", (0.0, 100.0), default=1.0)
        max_delta_step = Float("max_delta_step", (0.0, 100.0), default=0.0)
        subsample = Float("subsample", (0.00001, 1.0), default=1.0)
        #sampling_method = Categorical("sampling_method", ["uniform", "gradient_based"], default="uniform")
        colsample_bytree = Float("colsample_bytree", (0.00001, 1.0), default=1.0)
        colsample_bylevel = Float("colsample_bylevel", (0.00001, 1.0), default=1.0)
        colsample_bynode = Float("colsample_bynode", (0.00001, 1.0), default=1.0)
        reg_lambda = Float("lambda", (0, 100.0), default=1.0)
        alpha = Float("alpha", (0, 100.0), default=0.0)
        tree_method = Categorical("tree_method", ["auto", "exact", "approx", "hist"], default="auto")
        num_boost_round = Integer("num_boost_round", (1, 500), default=50)
        cs.add_hyperparameters([eta, gamma, max_depth, min_child_weight, max_delta_step, subsample,
                                colsample_bytree, colsample_bylevel, colsample_bynode, reg_lambda, alpha, tree_method, num_boost_round])
        return cs

    def train(self, config: Configuration, seed: int) -> float:
        params_dict = config.get_dictionary()
        num_boost_round = params_dict["num_boost_round"]
        del params_dict["num_boost_round"]

        costs = []
        for i in range(self.nfold):
            booster = xgb.train(params_dict, dtrain = self.Xys[i], num_boost_round = num_boost_round, obj=self.def_fn.get_obj_fn())
            yvalpred = booster.inplace_predict(self.valdatas[i][0])
            if len(self.auxdatas) > 0:
                valdl = self.prob.dec_loss(yvalpred, self.valdatas[i][1], aux_data=self.auxdatas[i])
            else:
                valdl = self.prob.dec_loss(yvalpred, self.valdatas[i][1])
            costs.append(valdl.mean())

        return np.mean(costs)



def compute_stderror(vec: np.ndarray) -> float:
    popstd = vec.std()
    n = len(vec)
    return (popstd * np.sqrt(n / (n - 1.0))) / np.sqrt(n)

def test_config_vec(params, prob, xgb_params, cusloss, xtrain, ytrain, xtest, ytest, auxtest):
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train(xgb_params, dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())
    testpred = booster.inplace_predict(xtest)
    itertestsmac = prob.dec_loss(testpred, ytest, aux_data=auxtest).flatten()
    return booster, itertestsmac

def test_config(params, prob, xgb_params, cusloss, xtrain, ytrain, xtest, ytest, auxtest):
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train(xgb_params, dtrain = Xy, num_boost_round = params["search_estimators"], obj = cusloss.get_obj_fn())
    testpred = booster.inplace_predict(xtest)
    itertestsmac = prob.dec_loss(testpred, ytest, aux_data=auxtest).flatten()
    return booster, itertestsmac.mean(), compute_stderror((itertestsmac))

def test_dir_weight(params, prob, xtrain, ytrain, xtest, ytest, auxtest):
    Xy = xgb.DMatrix(xtrain, ytrain)
    weight_vec = np.array([params["param_def"] for _  in range(2 * ytrain.shape[1])])
    dir_loss = search_weights_directed_loss(weight_vec)
    booster = xgb.train({"tree_method": params["tree_method"], "num_target": ytrain.shape[1]},
                        dtrain = Xy,
                        num_boost_round = params["search_estimators"],
                        obj = dir_loss.get_obj_fn())
    testpred = booster.inplace_predict(xtest)
    itertestsmac = prob.dec_loss(testpred, ytest, aux_data=auxtest).flatten()
    return booster, itertestsmac.mean(), compute_stderror((itertestsmac))

def test_weightmse(params, prob, xtrain, ytrain, xtest, ytest, auxtest):
    Xy = xgb.DMatrix(xtrain, ytrain)
    weight_vec = np.array([params["param_def"] for _  in range(ytrain.shape[1])])
    dir_loss = search_weights_loss(weight_vec)
    booster = xgb.train({"tree_method": params["tree_method"], "num_target": ytrain.shape[1]},
                        dtrain = Xy,
                        num_boost_round = params["search_estimators"],
                        obj = dir_loss.get_obj_fn())
    testpred = booster.inplace_predict(xtest)
    itertestsmac = prob.dec_loss(testpred, ytest, aux_data=auxtest).flatten()
    return booster, itertestsmac.mean(), compute_stderror((itertestsmac))

def test_reg(params, prob, xtrain, ytrain, xtest, ytest, auxtest):
    Xy = xgb.DMatrix(xtrain, ytrain)
    tree = xgb.XGBRegressor(tree_method=params["tree_method"], n_estimators=params["search_estimators"])
    tree.fit(xtrain, ytrain)
    testpred = tree.predict(xtest)
    itertestsmac = prob.dec_loss(testpred, ytest, aux_data=auxtest).flatten()
    return tree.get_booster(), itertestsmac.mean(), compute_stderror((itertestsmac))


def test_reg_lgb(params, prob, xtrain, ytrain, xtest, ytest, auxtest):
    lgb_model = lgb.LGBMRegressor(n_estimators=100)
    lgb_multi = MultiOutputRegressor(lgb_model)
    lgb_multi.fit(xtrain, ytrain)
    testpred = lgb_multi.predict(xtest)
    itertestsmac = prob.dec_loss(testpred, ytest, aux_data=auxtest).flatten()
    return lgb_multi, itertestsmac.mean(), compute_stderror((itertestsmac))


def test_square_log(params, prob, xtrain, ytrain, xtest, ytest, auxtest):
    '''Following code is from https://xgboost.readthedocs.io/en/stable/python/examples/custom_rmsle.html#sphx-glr-python-examples-custom-rmsle-py'''
    '''Train using Python implementation of Squared Log Error.'''
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        y = dtrain.get_label()
        return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for squared log error.'''
        y = dtrain.get_label()
        return ((-np.log1p(predt) + np.log1p(y) + 1) /
                np.power(predt + 1, 2))

    def squared_log(predt: np.ndarray,
                    dtrain: xgb.DMatrix):
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.

        :math:`\frac{1}{2}[log(pred + 1) - log(label + 1)]^2`

        '''
        predt = predt.flatten()
        predt[predt < -1] = -1 + 1e-6
        grad = gradient(predt, dtrain)
        hess = hessian(predt, dtrain)
        return grad, hess

    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train({"tree_method": params["tree_method"], "num_target": ytrain.shape[1]},
              dtrain=Xy,
              num_boost_round=params["search_estimators"],
              obj=squared_log)
    testpred = booster.inplace_predict(xtest)
    itertestsmac = prob.dec_loss(testpred, ytest, aux_data=auxtest).flatten()
    return booster, itertestsmac.mean(), compute_stderror((itertestsmac))


def smac_search_lgb(params, prob, model, n_trials, xtrain, ytrain, xtest, ytest, auxtest, test_history=False):
    scenario = Scenario(model.configspace, n_trials=n_trials)
    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)
    smac = HPOFacade(scenario, model.train_lgb, intensifier=intensifier, overwrite=True)
    records = []
    start_time = time.time()
    for cnt in range(n_trials):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train_lgb(info.config, seed=info.seed)
        value = TrialValue(cost=cost)
        records.append((value.cost, info.config))

        smac.tell(info, value)

        if test_history:
            _, testdl, teststderr = test_config(params, prob, model.get_xgb_params(), model.get_loss_fn(info.config), xtrain, ytrain, xtest, ytest, auxtest)
            print(f"Vec {model.get_vec(info.config)}")
            print(f"history test teststderr, {cost}, {testdl}, {teststderr}")

    print(f"Search takes {time.time() - start_time} seconds")
    records.sort()
    incumbent = records[0][1]
    params_vec = model.get_vec(incumbent)
    print(f"Seaerch Choose {params_vec}")
    cusloss = model.get_loss_fn(incumbent)
    return cusloss


def eval_config_lgb(params, prob, xgb_params, cusloss, xtrain, ytrain, xval, xtest):
    xy_lgb = lgb.Dataset(xtrain, ytrain)
    lgb_model = lgb.train({"boosting_type": "gbdt", "objective": cusloss.get_obj_fn()}, xy_lgb, num_boost_round=self.params["search_estimators"])

    trainsmacpred = lgb_model.predict(xtrain)
    valsmacpred = lgb_model.predict(xval)
    testsmacpred = lgb_model.predict(xtest)

    return lgb_model, trainsmacpred, valsmacpred, testsmacpred

def eval_xgb_hyper(params, loss_fn, prob, xtrain, ytrain, xtest, ytest, auxtest):
    if params["test_hyper"] == "hyperonly":
        model = XGBHyperSearch(prob, xtrain, ytrain)
    elif params["test_hyper"] == "hyperwdef":
        model = XGBHyperSearchwDefault(prob, xtrain, ytrain, params["param_low"], params["param_upp"], params["param_def"])

    scenario = Scenario(
        model.configspace,
        n_trials=params["n_trials"],  # We want to run max 50 trials (combination of config and seed)
    )

    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)
    smac = HPOFacade(scenario, model.train, intensifier=intensifier, overwrite=True)
    records = []

    for cnt in range(params["n_trials"]):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost)
        records.append((value.cost, info.config))

        smac.tell(info, value)

    # records.sort()
    # TODO check here '<' not supported between instances of 'Configuration' and 'Configuration'
    candidates = sorted(records, key=lambda x : x[0])
    select = len(candidates) - 1
    for i in range(1, len(candidates)):
        if candidates[i][0] != candidates[0][0]:
            select = i
            print(f"from idx 0 to {select} has the same cost randomly pick one")
            break
    idx = random.randint(0, select - 1)
    incumbent = records[idx][1]
    params_dict = incumbent.get_dictionary()
    print(f"Hyper final:{params_dict}")
    num_boost_round = params_dict["num_boost_round"]
    del params_dict["num_boost_round"]

    booster = xgb.train(params_dict, dtrain = xgb.DMatrix(xtrain, ytrain), num_boost_round=num_boost_round)
    ytestpred = booster.inplace_predict(xtest)
    hypertest = prob.dec_loss(ytestpred, ytest, aux_data=auxtest).flatten()
    return booster, hypertest




