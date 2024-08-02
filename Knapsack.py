# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Basedline Adapted code from LANCER
# From https://github.com/facebookresearch/LANCER

import os
import numpy as np
from PThenO import PThenO
from pyomo import environ as pe
from pyomo import opt as po
from utils import nn_utils


class KnapsackProblem(PThenO):
    def __init__(self, num_feats, weights, cap, num_items, kdim, n_cpus=1, solver="scip"):
        super(KnapsackProblem, self).__init__()
        self.num_feats = num_feats
        self.weights = np.array(weights)
        self.capacity = np.array([cap] * kdim)
        # changing capacity for minimization problem
        self.capacity = np.sum(self.weights, axis=1) - self.capacity
        self.items = list(range(self.weights.shape[1]))

        self.num_items = num_items  # dim of the cost vector
        self.kdim = kdim
        self.n_cpus = n_cpus

        _model, _vars = self.build_model()
        self._model = _model
        self._vars = _vars
        self._solverfac = po.SolverFactory(solver)

    def build_model(self, **kwargs):
        """
        A method to a SCIP model
        Returns:
            tuple: optimization model and variables
        """
        m = pe.ConcreteModel("knapsack")
        m.its = pe.Set(initialize=self.items)
        x = pe.Var(m.its, domain=pe.Binary)
        m.x = x
        # constraints
        m.cons = pe.ConstraintList()
        for i in range(len(self.capacity)):
            m.cons.add(sum(self.weights[i, j] * x[j] for j in self.items) >= self.capacity[i])
        m.obj = pe.Objective(sense=pe.minimize, expr=0)
        return m, x

    def _solve_single_core(self, z):
        self._model.del_component(self._model.obj)
        obj = sum(z[j] * self._vars[k] for j, k in enumerate(self._vars))
        self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
        self._solverfac.solve(self._model)
        sol = [pe.value(self._vars[k]) for k in self._vars]
        return sol

    def dec_loss(self, z_pred: np.ndarray, z_true: np.ndarray, **kwargs) -> np.ndarray:
        assert z_pred.shape == z_true.shape
        assert self.n_cpus > 0
        n_cpus = os.cpu_count() if self.n_cpus == -1 else self.n_cpus
        N = z_true.shape[0]
        f_hat_list = np.zeros((N, 1))

        for i in range(N):
            if i % 100 == 0:
                print("Solving MIP:", i)

            sol = self._solve_single_core(z_pred[i])
            f_hat_i_cp = np.dot(sol, z_true[i])
            f_hat_list[i, 0] = f_hat_i_cp


        return f_hat_list

    def generate_dataset_nn(self, N, rnd_nn, noise_width=0.0):
        # cost vectors
        z = np.random.uniform(0, 5, (N, self.num_items))
        c_tensor = torch.from_numpy(z)
        y_tensor = rnd_nn(c_tensor)
        y = y_tensor.cpu().detach().to_numpy()
        # noise
        epsilon = np.random.uniform(1 - noise_width, 1 + noise_width, self.num_feats)
        y_noisy = y * epsilon
        return y_noisy, z

    #def get_c_shapes(self):
    #    return self.num_items, self.num_items

    def get_modelio_shape(self):
        # TODO

        return self.num_feats, self.d

    def get_output_activation(self):
        return "relu"


if __name__ == "__main__":
    # For debug test
    # Based on https://github.com/facebookresearch/LANCER/blob/e11b16cf4451d5223d135f7a35c42fdb45e7ae47/DFL/scripts/run_lancer_dfl.py#L32
    cap, num_items, kdim, p = 45, 100, 5, 256
    weights = np.random.uniform(0, 1, (kdim, num_items))
    # ILP for multidimnesional knapsack
    bb_problem = KnapsackProblem(num_feats=p, weights=weights, cap=cap, num_items=num_items,
                                                    kdim=kdim)
    from lancer_learner import build_mlp
    rnd_nn = nn_utils.build_mlp(input_size=num_items, output_size=p, n_layers=1, size=500,
                                            activation="relu", output_activation="tanh")
    Y, Z = bb_problem.generate_dataset_nn(N=1000+1000, rnd_nn=rnd_nn, noise_width=0.1)
    Y_train, Y_test, Z_train, Z_test = train_test_split(Y, Z, test_size=1000, random_state=seed)
