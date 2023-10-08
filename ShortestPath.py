# Adapted code from LANCER
# From https://github.com/facebookresearch/LANCER

import numpy as np
from pyomo import environ as pe
from pyomo import opt as po
from PThenO import PThenO
from sklearn.model_selection import train_test_split
import torch

class ShortestPath(PThenO):
    def __init__(self,
                 num_feats,
                 grid,
                 num_train,
                 num_val,
                 num_test,
                 seed,
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


        X, Y = self.generate_dataset(N=num_train + num_val + num_test, deg=6, noise_width=0.5)
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=num_test, random_state=seed)

        self.X_train = torch.tensor(X_trainval[:num_train]).float()
        self.Y_train = torch.tensor(Y_trainval[:num_train]).float()

        self.X_val = torch.tensor(X_trainval[num_train:]).float()
        self.Y_val = torch.tensor(Y_trainval[num_train:]).float()

        self.X_test = torch.tensor(X_test).float()
        self.Y_test = torch.tensor(Y_test).float()

        self.rand_seed = seed * 17
        self._set_seed(self.rand_seed)

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
                elif i == self.m - 1 and j == self.m - 1:
                    m.cons.add(expr == 1)
                # transition
                else:
                    m.cons.add(expr == 0)
        m.obj = pe.Objective(sense=pe.minimize, expr=0)
        return m, x

    def predeval(self, z_pred: np.ndarray, z_true: np.ndarray, verbose=False, **kwargs) -> np.ndarray:
        # Z in lancer paper means y in lodl
        assert z_pred.shape == z_true.shape
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

    def get_train_data(self, **kwargs):
        return self.X_train, self.Y_train,  [None for _ in range(len(self.X_train))]

    def get_test_data(self, **kwargs):
        return self.X_test, self.Y_test,  [None for _ in range(len(self.X_test))]

    def get_twostageloss(self, **kwargs):
        return "mse"

    def get_val_data(self, **kwargs):
        return self.X_val, self.Y_val,  [None for _ in range(len(self.X_val))]


if __name__ == "__main__":
    sp = ShortestPath(num_feats=5, grid=(5, 5), num_train=800, num_val=200, num_test=1000, seed=1)
    import pdb

    X_val, Y_val, _ = sp.get_val_data()
    zs = sp.get_decision(Y_val)

    zobjs = sp.get_objective(Y_val, zs)
    zobjs_eval = sp.predeval(Y_val.detach().numpy(), Y_val.detach().numpy())
    pdb.set_trace()

