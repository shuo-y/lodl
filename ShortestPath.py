# Adapted code from LANCER
# From https://github.com/facebookresearch/LANCER

import numpy as np
from pyomo import environ as pe
from pyomo import opt as po
from PThenO import PThenO
from sklearn.model_selection import train_test_split
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class ShortestPath(PThenO):
    def __init__(self,
                 num_feats,
                 grid,
                 solver="glpk"):
        super(ShortestPath, self).__init__()
        self.num_feats = num_feats
        self.m = grid[0]
        self.n = grid[1]
        # dimension of the cost vector
        self.d = (self.m - 1) * self.n + (self.n - 1) * self.m
        self.arcs = self._get_arcs()
        _model, _vars = self.build_model()
        self._model = _model
        self._vars = _vars
        self._solverfac = po.SolverFactory(solver)
        self._cvxlayer = self._build_cvxpylayer()


    def _get_arcs(self):
        """
        A helper method to get list of arcs for grid network
        """
        arcs = []
        for i in range(self.m):
            # edges on rows
            for j in range(self.n - 1):
                v = i * self.n + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.m - 1:
                continue
            for j in range(self.n):
                v = i * self.n + j
                arcs.append((v, v + self.n))
        return arcs

    def build_model(self, **kwargs):
        """
        A method to build pyomo model: Linear Program
        Returns:
            tuple: optimization model and variables
        """
        m = pe.ConcreteModel(name="shortest path")
        x = pe.Var(self.arcs, name="x", within=pe.NonNegativeReals)
        m.x = x
        m.cons = pe.ConstraintList()

        for i in range(self.m):
            for j in range(self.n):
                v = i * self.n + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[e]
                    # flow out
                    elif v == e[0]:
                        expr -= x[e]
                # source
                if i == 0 and j == 0:
                    m.cons.add(expr == -1)
                # sink
                elif i == self.m - 1 and j == self.n - 1:
                    m.cons.add(expr == 1)
                # transition
                else:
                    m.cons.add(expr == 0)
        m.obj = pe.Objective(sense=pe.minimize, expr=0)
        return m, x

    def _build_cvxpylayer(self, **kwargs):
        """
        A model same as build_model but use cvxpy
        """
        z = cp.Parameter(len(self.arcs))
        x = cp.Variable(len(self.arcs))
        cons = []

        for i in range(self.m):
            for j in range(self.n):
                v = i * self.n + j
                expr = 0
                for ind, e in enumerate(self.arcs):
                    # flow in
                    if v == e[1]:
                        expr += x[ind]
                    # flow out
                    elif v == e[0]:
                        expr -= x[ind]
                # source
                if i == 0 and j == 0:
                    cons.append(expr == -1)
                # sink
                elif i == self.m - 1 and j == self.m - 1:
                    cons.append(expr == 1)
                # transition
                else:
                    cons.append(expr == 0)

        cons.append(x >= 0)
        objective = cp.Minimize(x.T @ z)
        problem = cp.Problem(objective, cons)
        return CvxpyLayer(problem, parameters=[z], variables=[x])




    def _dec_loss_cvxpylayer(self, z_pred: np.ndarray, z_true: np.ndarray, verbose=False, **kwargs) -> np.ndarray:
        assert z_pred.shape == z_true.shape
        if verbose:
            print("Solve using cvxpylayer")
        N = z_true.shape[0]
        f_hat_list = []

        zpt = torch.tensor(z_pred)
        xdec = self._cvxlayer(zpt)[0]

        f_hat_list = (xdec * z_true).sum(dim=1).unsqueeze(1)
        return f_hat_list.detach().clone().numpy()


    def dec_loss(self, z_pred: np.ndarray, z_true: np.ndarray, verbose=False, **kwargs) -> np.ndarray:
        # Z in lancer paper means y in lodl
        assert z_pred.shape == z_true.shape
        if "use_cvxpylayer" in kwargs and kwargs["use_cvxpylayer"] == True:
            return self._dec_loss_cvxpylayer(z_pred, z_true, verbose)

        N = z_true.shape[0]
        f_hat_list = []
        # TODO: run this loop in parallel
        for i in range(N):
            if i%100 == 0 and verbose:
                print("Solving LP:", i, " out of ", N)
            self._model.del_component(self._model.obj)
            obj = sum(z_pred[i, j] * self._vars[k] for j, k in enumerate(self._vars))
            self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
            self._solverfac.solve(self._model)
            sol = [pe.value(self._vars[k]) for k in self._vars]
            #print(sol)
            ############################
            f_hat_i_cp = np.dot(sol, z_true[i])
            f_hat_list.append([f_hat_i_cp])
        return np.array(f_hat_list)

    def generate_dataset(self, N, deg=1, noise_width=0):
        """
        Generate synthetic dataset for the DFL shortest path problem
        :param N: number of points
        :param deg: degree of polynomial to enforce nonlinearity
        :param noise_width: add eps noise to the cost vector
        :return: dataset of features x and the ground truth cost vector of edges c
        """
        # random matrix parameter B
        B = np.random.binomial(1, 0.5, (self.d, self.num_feats))
        # feature vectors
        x = np.random.normal(0, 1, (N, self.num_feats))
        # cost vectors
        z = np.zeros((N, self.d))
        for i in range(N):
            # cost without noise
            zi = (np.dot(B, x[i].reshape(self.num_feats, 1)).T / np.sqrt(self.num_feats) + 3) ** deg + 1
            # rescale
            zi /= 3.5 ** deg
            # noise
            epislon = np.random.uniform(1 - noise_width, 1 + noise_width, self.d)
            zi *= epislon
            z[i, :] = zi
        return x, z

    def get_decision(self, Y, **kwargs):
        if Y.ndim == 1:
            # A single item
            assert len(Y) == self.Y_train.shape[1]
            y_pred = Y.detach().numpy()
            self._model.del_component(self._model.obj)
            obj = sum(y_pred[j] * self._vars[k] for j, k in enumerate(self._vars))
            self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
            self._solverfac.solve(self._model)
            sol = [pe.value(self._vars[k]) for k in self._vars]
            return torch.tensor(sol).float()
        elif Y.ndim == 2:
            # Solve for multiple instances
            N = len(Y)
            assert Y.shape[1] == self.Y_train.shape[1]
            y_preds = Y.detach().numpy()
            all_sols = []
            for i in range(N):
                self._model.del_component(self._model.obj)
                obj = sum(y_preds[i, j] * self._vars[k] for j, k in enumerate(self._vars))
                self._model.obj = pe.Objective(sense=pe.minimize, expr=obj)
                self._solverfac.solve(self._model)
                sol = [pe.value(self._vars[k]) for k in self._vars]
                all_sols.append(sol)
            return torch.tensor(all_sols).float()
        else:
            assert "Not supported Y shape"

    def get_objective(self, Y, Z, **kwargs):
        N = np.prod(Y.size()) // Y.shape[-1]
        dim = Y.shape[-1]

        return -torch.bmm(Y.view(N, 1, dim), Z.view(N, dim, 1)).squeeze()

    def get_modelio_shape(self):
        # This one is less than 3 dimensions
        # xgboost coupled or decoupled are the same
        # for neural network just use dense
        # dense_coupled is for 3 dimensions
        return self.num_feats, self.d

    def get_output_activation(self):
        return "relu"

    """
    def get_train_data(self, **kwargs):
        return self.X_train, self.Y_train,  [None for _ in range(len(self.X_train))]

    def get_test_data(self, **kwargs):
        return self.X_test, self.Y_test,  [None for _ in range(len(self.X_test))]

    def get_twostageloss(self, **kwargs):
        return "mse"

    def get_val_data(self, **kwargs):
        return self.X_val, self.Y_val,  [None for _ in range(len(self.X_val))]
    """


if __name__ == "__main__":
    sp = ShortestPath(num_feats=5, grid=(5, 5))
    x, z = sp.generate_dataset(100)
    dloss1 = sp.dec_loss(z, z, verbose=True)
    dloss2 = sp._dec_loss_cvxpylayer(z, z, verbose=True)
    import pdb
    pdb.set_trace()

