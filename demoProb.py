# Demo prediction than optimize
# From paper Perils of Learning
import random
import numpy as np
import cvxpy as cp
from PThenO import PThenO

def optsingleprod(ypred, y_true, **kwargs):
    def opt(yi):
        # A demo predict than optimize
        # min y0 y1 c
        # c \in {-1, 1}
        if yi[0] * yi[1] >= 0:
            c = -1
        else:
            c = 1
        return c

    def prod(yi):
        return yi[0] * yi[1]

    dec = np.apply_along_axis(opt, 1, y_pred)
    # How if using single decision
    obj = np.apply_along_axis(np.prod, 1, y_true) * dec
    return obj.reshape(obj.shape[0], 1)

def opttwoprod(y_pred, y_true, **kwargs):
    """
    From paper  Perils of Learning Before Optimizing
    """
    def prodfirst2(yi):
        return yi[0] * yi[1]

    def prodlast2(yi):
        return yi[2] * yi[3]

    predfirst2 = np.apply_along_axis(prodfirst2, 1, y_pred)
    predlast2 = np.apply_along_axis(prodlast2, 1, y_pred)

    if predfirst2.mean() <= predlast2.mean():
        c = [1, 0]
    else:
        c = [0, 1]

    # How if using single decision

    truefirst2 = np.apply_along_axis(prodfirst2, 1, y_true)
    truelast2 = np.apply_along_axis(prodlast2, 1, y_true)

    return c[0] * truefirst2.mean() + c[1] * truelast2.mean()

def gen_xy_twoprod(num_ins, dnum, N, epsilon, num_feat, **kwargs):
    """
    num_ins  number of optimization instances
    nd number of d in each instance in paper
    The Perils of Learning
    """
    Ymat = np.zeros((num_ins, dnum, 4))

    for i in range(num_ins):
        arr = [0 if d < dnum // 2 else 1 for d in range(dnum)]
        random.shuffle(arr)
        for d in range(dnum):
            Ymat[i][d][0] = float(arr[d]) * N
            Ymat[i][d][1] = float(1 - arr[d]) * N
            Ymat[i][d][2] = 0.5 * N - epsilon
            Ymat[i][d][3] = 0.5 * N - epsilon

    xmat = np.random.random((num_ins, dnum, num_feat))
    if "xfeat" in kwargs and kwargs["xfeat"] == "xcons":
        xmat = np.ones((num_ins, dnum, num_feat))

    return xmat, Ymat


## 2-dimensional Rosenbrock function https://automl.github.io/SMAC3/v2.0.2/examples/1_basics/3_ask_and_tell.html
class ProdObj(PThenO):
    def __init__(self, optdclossfn, numD):
        """
        optfn is for the optimization functions
        genfn is for generating features of the dataset
        """
        super(ProdObj, self).__init__()
        self.num_feats = None
        self.optdclossfn = optdclossfn
        self.numD = numD

    def dec_loss_single(self, y_pred: np.ndarray, y_true: np.ndarray, verbose=False, **kwargs) -> float:
        return self.optdclossfn(y_pred, y_true)

    def dec_loss(self, y_pred: np.ndarray, y_true: np.ndarray, verbose=False, **kwargs) -> np.ndarray:
        return_dec = True if "return_dec" in kwargs and kwargs["return_dec"] == True else False
        res = []
        assert len(y_pred) == len(y_true)

        if return_dec:
            decs = []

        if y_pred.ndim == 2:
            numIns = len(y_pred)
            y_pred = y_pred.reshape(numIns // self.numD, self.numD, 4)
            y_true = y_true.reshape(numIns // self.numD, self.numD, 4)

        for i in range(len(y_pred)):
            # <<<<<<< Solve the optimization problem it is based on
            # paper Perils of ... and generated from ChatGPT
            # Decision variables (z_i1, z_i2)
            d = y_pred.shape[1]
            z_1 = cp.Variable(d)
            z_2 = cp.Variable(d)

            y = y_pred[i]

            # Constant C (can be set to any constant value)
            C = 10

            # Objective function
            objective = cp.Minimize(C + cp.sum([y[di, 0] * y[di, 1] * z_1[di] + y[di, 2] * y[di, 3] * z_2[di] for di in range(d)]))

            # Constraints
            constraints = [
                cp.sum(z_1 + z_2) >= d,  # Sum constraint
                z_1 >= 0,  # z_i1 >= 0
                z_2 >= 0,  # z_i2 >= 0
                z_1 <= 1,  # z_i1 <= 1
                z_2 <= 1,  # z_i2 <= 1
            ]

            # Define and solve the optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if return_dec:
                decs.append([[z_1.value[di], z_2.value[di]]  for di in range(d)])

            obj = C + sum([y_true[i][di, 0] * y_true[i][di, 1] * z_1.value[di] + y_true[i][di, 2] * y_true[i][di, 3] * z_2.value[di]   for di in range(d)])
            # >>>>>>>
            res.append(obj)

        if return_dec:
            return np.array(decs), np.array(res)
        return np.array(res)


    def rand_loss(self, z_true: np.ndarray) -> np.ndarray:
        rand_dec = np.random.randint(2, size=len(z_true))
        rand_dec = rand_dec * 2 - 1
        obj = np.apply_along_axis(np.prod, 1, z_true) * rand_dec
        return obj

    def generate_feat(self, N, deg, noise_width, num_feats, y):
        dim = len(y[0])
        B = np.random.binomial(1, 0.5, (num_feats, dim))
        x = np.random.normal(0, 1, (N, num_feats))
        for i in range(N):
            # cost without noise

            xi = (np.dot(B, y[i].reshape(dim, 1)).T / np.sqrt(num_feats) + 3) ** deg + 1
            # rescale
            xi /= 3.5 ** deg
            # noise
            epislon = np.random.uniform(1 - noise_width, 1 + noise_width, num_feats)
            xi *= epislon
            x[i, :] = xi
        return x

    def generate_dataset_ind(self, N, deg, noise_width, num_feats, dim, mus, sigs):
        self.num_feats = num_feats
        z = np.zeros((N, dim))
        for d in range(dim):
            z[:,d] = np.random.normal(mus[d], sigs[d], N)

        B = np.random.binomial(1, 0.5, (num_feats, dim))
        # feature vectors
        x = np.random.normal(0, 1, (N, num_feats))

        for i in range(N):
            # cost without noise

            xi = (np.dot(B, z[i].reshape(dim, 1)).T / np.sqrt(num_feats) + 3) ** deg + 1
            # rescale
            xi /= 3.5 ** deg
            # noise
            epislon = np.random.uniform(1 - noise_width, 1 + noise_width, num_feats)
            xi *= epislon
            x[i, :] = xi

        return x, z

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
        self.num_feats = num_feats
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

    def generate_dataset_uniform(self, N,
                         num_feats, d=2, mean=np.array([-0.9, -0.9]),
                         cov=np.array([[1, -1], [-1, 5]])):
        """
        From the LANCER code
        Generate synthetic dataset for the DFL shortest path problem
        :param N: number of points
        :return: dataset of features x and z
        """
        self.num_feats = num_feats
        # feature vectors
        x = np.random.uniform(0, 1, (N, num_feats))
        # cost vectors
        z = np.random.multivariate_normal(mean, cov, N)

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
        return self.num_feats, 2

    def get_objective(self, y_vec: np.ndarray, dec: np.ndarray, **kwargs):
        return np.apply_along_axis(np.prod, 1, y_vec) * dec

    def get_output_activation(self):
        pass

