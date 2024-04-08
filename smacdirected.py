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
from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss, search_quadratic_loss
from PThenO import PThenO

# QuadLoss is based on SMAC examples https://automl.github.io/SMAC3/v2.0.2/examples/1_basics/2_svm_cv.html
class DirectedLoss:
    def __init__(self, prob, params, xtrain, ytrain, xval, yval, valtruedl, auxdata):
        self.Xy = xgb.DMatrix(xtrain, ytrain)
        self.xval = xval
        self.yval = yval
        self.params = params
        self.valtruedl = valtruedl
        self.prob = prob
        self.aux_data = auxdata
        self.loss_fn = search_weights_directed_loss


    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        configs = [Float(f"w{i}", (0.01, 100), default=1) for i in range(self.yval.shape[1])]
        cs.add_hyperparameters(configs)
        return cs

    def train(self, configs: Configuration, seed: int) -> float:
        configarray = [configs[f"w{i}"] for i in range(self.yval.shape[1])]
        weight_vec = np.array(configarray)

        cusloss = search_weights_directed_loss(self.yval.shape[1], weight_vec)
        booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]},
                             dtrain = self.Xy, num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

        yvalpred = booster.inplace_predict(self.xval)
        valdl = self.prob.dec_loss(yvalpred, self.yval, aux_data=self.aux_data)

        cost = (valdl - self.valtruedl).mean()
        return cost

    def get_vec(self, incumbent) -> np.ndarray:
        # Get the weight vector from incumbent
        arr = [incumbent[f"w{i}"] for i in range(self.yval.shape[1])]
        return np.array(arr)


class QuadSearch:
    def __init__(self, prob, params, xtrain, ytrain, xval, yval, valtruedl, auxdata):
        self.Xy = xgb.DMatrix(xtrain, ytrain)
        self.xval = xval
        self.yval = yval
        self.params = params
        self.valtruedl = valtruedl
        self.prob = prob
        self.aux_data = auxdata
        self.total_params_n = ((1 + self.yval.shape[1]) * self.yval.shape[1]) // 2
        self.loss_fn = search_quadratic_loss

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        configs = [Float(f"w{i}", (0.001, 1), default=1) for i in range(self.total_params_n)]
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

        cusloss = search_quadratic_loss(self.yval.shape[1], base_vec, 0.01)
        booster = xgb.train({"tree_method": self.params["tree_method"], "num_target": self.yval.shape[1]},
                             dtrain = self.Xy, num_boost_round = self.params["search_estimators"], obj = cusloss.get_obj_fn())

        yvalpred = booster.inplace_predict(self.xval)
        valdl = self.prob.dec_loss(yvalpred, self.yval, aux_data=self.aux_data)

        cost = (valdl - self.valtruedl).mean()
        return cost

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

