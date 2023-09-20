from numpy import square
import torch
from math import sqrt
from functools import reduce
import operator
import pdb
import numpy as np

from utils import View


# TODO: Pretty it up
def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=10,
    activation='relu',
    output_activation='sigmoid',
):
    if num_layers > 1:
        if intermediate_size is None:
            intermediate_size = max(num_features, num_targets)
        if activation == 'relu':
            activation_fn = torch.nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = torch.nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [torch.nn.Linear(num_features, intermediate_size), activation_fn()]
        for _ in range(num_layers - 2):
            net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        if not isinstance(num_targets, tuple):
            net_layers.append(torch.nn.Linear(intermediate_size, num_targets))
        else:
            net_layers.append(torch.nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1)))
            net_layers.append(View(num_targets))
    else:
        if not isinstance(num_targets, tuple):
            net_layers = [torch.nn.Linear(num_features, num_targets)]
        else:
            net_layers = [torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1)), View(num_targets)]

    if output_activation == 'relu':
        net_layers.append(torch.nn.ReLU())
    elif output_activation == 'sigmoid':
        net_layers.append(torch.nn.Sigmoid())
    elif output_activation == 'tanh':
        net_layers.append(torch.nn.Tanh())
    elif output_activation == 'softmax':
        net_layers.append(torch.nn.Softmax(dim=-1))

    return torch.nn.Sequential(*net_layers)

class NNCoupled(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        num_layers=3,
        intermediate_size=10,
        activation='relu',
        output_activation='sigmoid'
    ):
        """
        Here the num_features and num_targets should be tuple
        """
        super(NNCoupled, self).__init__()

        if activation == 'relu':
            activation_fn = torch.nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = torch.nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))

        if (num_layers==1) :
            net_layers = [torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1))]
        else:
            input_dim = reduce(operator.mul, num_targets, 1)
            output_dim = reduce(operator.mul, num_targets, 1)

            net_layers = [torch.nn.Linear(input_dim, intermediate_size)]

            for i in range(num_layers - 2):
                net_layers.append(activation_fn())
                net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))

            net_layers.append(activation_fn())
            net_layers.append(torch.nn.Linear(intermediate_size, output_dim))

        if output_activation == 'relu':
            net_layers.append(torch.nn.ReLU())
        elif output_activation == 'sigmoid':
            net_layers.append(torch.nn.Sigmoid())
        elif output_activation == 'tanh':
            net_layers.append(torch.nn.Tanh())
        elif output_activation == 'softmax':
            net_layers.append(torch.nn.Softmax(dim=-1))

        self.net = torch.nn.Sequential(*net_layers)
        self.output_shape = num_targets

    def forward(self, inp):
        if inp.ndim == 2:
            # This is one single item
            x = inp.flatten()
            y = self.net(x)
            output = y.reshape(self.output_shape)
        elif inp.ndim == 3:
            # First dim is batch size
            x = inp.flatten(start_dim=1)
            y = self.net(x)
            output = y.reshape(inp.shape[0], *self.output_shape)
        return output

class DenseLoss(torch.nn.Module):
    """
    A Neural Network-based loss function
    """

    def __init__(
        self,
        Y,
        num_layers=4,
        hidden_dim=100,
        activation='relu'
    ):
        super(DenseLoss, self).__init__()
        # Save true labels
        self.Y = Y.detach().view((-1))
        # Initialise model
        self.model = torch.nn.Parameter(dense_nn(Y.numel(), 1, num_layers, intermediate_size=hidden_dim, output_activation=activation))

    def forward(self, Yhats):
        # Flatten inputs
        Yhats = Yhats.view((-1, self.Y.numel()))

        return self.model(Yhats)


class WeightedMSE(torch.nn.Module):
    """
    A weighted version of MSE
    """
    import numpy as np

    def __init__(self, Y, min_val=1e-3, magnify=1.0, weights_vec=[]):
        super(WeightedMSE, self).__init__()
        # Weights vec will ignore the min_val and magnify
        # Save true labels
        # Maybe refer to https://stackoverflow.com/questions/55959918/in-pytorch-how-to-i-make-certain-module-parameters-static-during-training
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.Y.requires_grad = False
        self.min_val = min_val

        self.magnify = magnify
        # Initialise paramters
        if len(weights_vec) == 0:
            self.weights = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        else:
            self.weights = torch.nn.Parameter(torch.tensor(weights_vec))
            self.min_val = self.weights.min().item()


    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))

        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = self.magnify * (squared_error * self.weights.clamp(min=self.min_val)).mean(dim=-1)

        return weighted_mse

    def my_grad_hess(self, yhat: np.ndarray, y: np.ndarray):
        """
        yhat should be a numpy array
        """
        #y = self.Y.detach().cpu().numpy()
        w = self.weights.clamp(min=self.min_val).detach().cpu().numpy()
        yhat = yhat.flatten()
        y = y.flatten()
        diff = (yhat - y)
        grad = 2 * self.magnify * (w * diff) / len(yhat)
        hess = 2 * self.magnify * w / len(yhat)
        return grad, hess

    def get_jnp_fun(self):
        import jax.numpy as jnp
        """
        Use jnp based on https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.square.html
        and some others like  https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.mean.html
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.multiply.html
        and
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.subtract.html
        https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.square.html
        """
        w = jnp.array(self.weights.clamp(min=self.min_val).detach().cpu().numpy())
        y = jnp.array(self.Y.detach().cpu().numpy())
        def jnp_forward(yhat):
            diff = yhat - y
            res = self.magnify * ((w * (diff ** 2)).mean())
            return res
        return jnp_forward




class WeightedMSEPlusPlus(torch.nn.Module):
    """
    A weighted version of MSE
    """
    def __init__(self, Y, min_val=1e-3):
        super(WeightedMSEPlusPlus, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.Y.requires_grad = False
        self.min_val = min_val

        # Initialise paramters
        self.weights_pos = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights_neg = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))

        # Get weights for positive and negative components separately
        pos_weights = (Yhat > self.Y.unsqueeze(0)).float() * self.weights_pos.clamp(min=self.min_val)
        neg_weights = (Yhat < self.Y.unsqueeze(0)).float() * self.weights_neg.clamp(min=self.min_val)
        weights = pos_weights + neg_weights

        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = (squared_error * weights).mean(dim=-1)

        return weighted_mse

    def get_jnp_fun(self):
        import jax.numpy as jnp

        posw = jnp.array(self.weights_pos.clamp(min=self.min_val).detach().cpu().numpy())
        negw = jnp.array(self.weights_neg.clamp(min=self.min_val).detach().cpu().numpy())
        y = jnp.array(self.Y.detach().cpu().numpy())

        def jnp_forward(yhat):
            cur_pos_w = (yhat > y) * posw
            cur_neg_w = (yhat < y) * negw
            w = cur_pos_w + cur_neg_w
            diff = (yhat - y)
            res = (w * (diff ** 2)).mean()
            return res
        return jnp_forward


    def my_grad_hess(self, yhat: np.ndarray, y: np.ndarray):
        yhat = yhat.flatten()
        posw = self.weights_pos.clamp(min=self.min_val).detach().cpu().numpy()
        negw = self.weights_neg.clamp(min=self.min_val).detach().cpu().numpy()

        y = y.flatten()

        diff = (yhat - y)
        posdiff = (diff > 0) * diff
        negdiff = (diff < 0) * diff

        grad = (2 * (posw * posdiff) + 2 * (negw * negdiff))/len(yhat)
        hess = (2 * (posw * (diff > 0)) + 2 * (negw * (diff < 0))) / len(yhat)
        return grad, hess


class WeightedCE(torch.nn.Module):
    """
    A weighted version of CE
    """

    def __init__(self, Y, min_val=1):
        super(WeightedCE, self).__init__()
        # Save true labels
        self.Y_raw = Y.detach()
        self.Y = self.Y_raw.view((-1))
        self.Y.requires_grad = False
        self.num_dims = self.Y.shape[0]
        self.min_val = min_val

        # Initialise paramters
        self.weights_pos = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights_neg = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhat):
        # Flatten inputs
        if len(self.Y_raw.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[:-len(self.Y_raw.shape)], self.num_dims))


        # Get weights for positive and negative components separately
        pos_weights = (Yhat > self.Y.unsqueeze(0)).float() * self.weights_pos.clamp(min=self.min_val)
        neg_weights = (Yhat < self.Y.unsqueeze(0)).float() * self.weights_neg.clamp(min=self.min_val)
        weights = pos_weights + neg_weights

        # Compute MSE
        error = torch.nn.BCELoss(reduction='none')(Yhat, self.Y.expand(*Yhat.shape))
        weighted_ce = (error * weights).mean(dim=-1)


        #weights_no_grad = torch.tensor(weights).flatten()
        #weights_no_grad.require_grad = False
        #error_2 = torch.nn.BCELoss(reduction='none', weight=weights_no_grad)(Yhat, self.Y.expand(*Yhat.shape))
        #print("use torch api", error_2.mean().item())
        #print("product", (error * weights_no_grad).mean())
        # results are the same but weights cannot require grad
        # print("debug wieghts", weights)

        return weighted_ce

    def my_grad_hess(self, yhat: np.ndarray, y: np.ndarray):
        import jax.numpy as jnp

        posw = jnp.array(self.weights_pos.clamp(min=self.min_val).detach().cpu().numpy())
        negw = jnp.array(self.weights_neg.clamp(min=self.min_val).detach().cpu().numpy())

        yhat = yhat.flatten()
        y = y.flatten()

        cur_pos_w = (yhat > y) * posw
        cur_neg_w = (yhat < y) * negw
        w = cur_pos_w + cur_neg_w

        grad = -(w * (y / yhat + (1 - y)/(yhat - 1)))/len(yhat)
        hess = -(w * (-y/(yhat ** 2) + (y - 1) / ((yhat - 1) ** 2))) / len(yhat)

        return grad, hess

    def get_jnp_fun(self):
        import jax.numpy as jnp

        posw = jnp.array(self.weights_pos.clamp(min=self.min_val).detach().cpu().numpy())
        negw = jnp.array(self.weights_neg.clamp(min=self.min_val).detach().cpu().numpy())
        y = jnp.array(self.Y.detach().cpu().numpy())

        def jnp_forward(yhat):
            cur_pos_w = (yhat > y) * posw
            cur_neg_w = (yhat < y) * negw
            w = cur_pos_w + cur_neg_w

            loss_val = - w * (y * jnp.clip(jnp.log(yhat), -100, 100)  + (1 - y) * jnp.clip(jnp.log(1 - yhat), -100, 100))
            loss_val = loss_val.mean()
            return loss_val

        return jnp_forward



class WeightedMSESum(torch.nn.Module):
    """
    A weighted version of MSE-Sum
    """

    def __init__(self, Y):
        super(WeightedMSESum, self).__init__()
        # Save true labels
        assert len(Y.shape) == 2  # make sure it's a multi-dimensional input
        self.Y = Y.detach()
        self.Y.requires_grad = False

        # Initialise paramters
        self.msesum_weights = torch.nn.Parameter(torch.rand(Y.shape[0]))

    def forward(self, Yhats):
        # Get weighted MSE-Sum
        sum_error = (self.Y - Yhats).mean(dim=-1)
        row_error = sum_error.square()
        weighted_mse_sum = (row_error * self.msesum_weights).mean(dim=-1)

        return weighted_mse_sum

    def my_grad_hess(self, yhat, y):
        #raise Exception("Not implmented yet!")
        #print(yhat - y)
        diff = (yhat - y).mean(axis=-1)
        w = self.msesum_weights.detach().numpy()
        grad = 2 * diff * w
        #print(diff)
        #print(w)
        grad = grad.repeat(self.Y.shape[1])
        grad = grad / (self.Y.shape[1] * self.Y.shape[0])
        hess = 2 * w
        hess = hess.repeat(self.Y.shape[1]) / (self.Y.shape[0] * (self.Y.shape[1]**2))
        return grad, hess

    def get_jnp_fun(self):
        """
        Get forward function for Jax autodiff
        https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        """
        import jax.numpy as jnp
        yn = jnp.array(self.Y.numpy())
        weights = jnp.array(self.msesum_weights.detach().numpy())
        def npforward(yhat):
            yhat = yhat.reshape(*yn.shape)
            diff = (yhat - yn).mean(axis=-1)
            res = (weights * (diff ** 2)).mean()
            return res
        return npforward


class TwoVarQuadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(self, Y):
        super(TwoVarQuadratic, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))

        # Initialise paramters
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.beta = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, Yhat):
        """
        """
        # Flatten inputs
        Yhat = Yhat.view((Yhat.shape[0], -1))

        # Difference of squares
        # Gives diagonal elements
        diag = (self.Y - Yhat).square().mean()

        # Difference of sum of squares
        # Gives cross-terms
        cross = (self.Y - Yhat).mean().square()

        return self.alpha * diag + self.beta * cross


class QuadraticPlusPlus(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self,
        Y,  # true labels
        quadalpha=1e-3,  # regularisation weight
        **kwargs
    ):
        super(QuadraticPlusPlus, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))
        self.Y.requires_grad = False
        self.num_dims = self.Y.shape[0]

        # Create quadratic matrices
        bases = torch.rand((self.num_dims, self.num_dims, 4)) / (self.num_dims * self.num_dims)
        self.bases = torch.nn.Parameter(bases)  

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[:-len(self.Y_raw.shape)], self.num_dims))

        # Measure distance between predicted and true distributions
        diff = (self.Y - Yhat).unsqueeze(-2)

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = self._get_basis(Yhat).clamp(-10, 10)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        # Add MSE as regularisation
        mse = diff.square().mean(dim=-1).squeeze()

        return quad + self.alpha * mse

    def _get_basis(self, Yhats):
        # Figure out which entries to pick
        #   Are you above or below the true label
        direction = (Yhats > self.Y).type(torch.int64)
        #   Use this to figure out the corresponding index
        direction_col = direction.unsqueeze(-1)
        direction_row = direction.unsqueeze(-2)
        index = (direction_col + 2 * direction_row).unsqueeze(-1)

        # Pick corresponding entries
        bases = self.bases.expand(*Yhats.shape[:-1], *self.bases.shape)
        basis = bases.gather(-1, index).squeeze()
        return torch.tril(basis)

    def my_grad_hess(self, yhat, y):
        yhat = yhat.flatten()
        # assume yhat and y are one dimensiona
        y = y.flatten()
        assert len(yhat.shape) == 1
        assert y.shape == yhat.shape
        direction = np.array((yhat > y), dtype=int)
        # use expand_dims based on https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
        # and also see previous code
        direction_col = np.expand_dims(direction, axis=1)
        direction_row = np.expand_dims(direction, axis=0)
        indices = np.expand_dims((direction_col + 2 * direction_row), axis=2)
        # use numpy take https://numpy.org/doc/stable/reference/generated/numpy.take.html
        # use take along axis similar to torch.gather https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html
        base = np.take_along_axis(self.bases.detach().numpy(), indices, axis=2).squeeze()
        # use tril based on https://numpy.org/doc/stable/reference/generated/numpy.tril.html
        base = np.tril(base)
        base = np.clip(base, -10, 10)
        
        diff = y - yhat
        hmat = base @ base.T
        hmat = hmat + hmat.T
        grad = - (hmat @ diff) - 2 * self.alpha * diff/len(y)
        hess = np.diagonal(hmat) + 2 * self.alpha / len(y)

        return grad, hess


    def get_jnp_fun(self, ret_naive=False):
        import jax.numpy as jnp
        basis = jnp.array(self.bases.detach().cpu().numpy())
        alpha = self.alpha
        y = jnp.array(self.Y.detach().cpu().numpy())
        def jnpforward(yhat):
            base = jnp.zeros((len(y), len(y)))
            for i in range(len(y)):
                for j in range(0, i + 1):
                    if yhat[i] > y[i] and yhat[j] > y[j]:
                        base = base.at[i, j].set(basis[i][j][3])
                    elif yhat[i] > y[i] and yhat[j] <= y[j]:
                        base = base.at[i, j].set(basis[i][j][1])
                    elif yhat[i] <= y[i] and yhat[j] > y[j]:
                        base = base.at[i, j].set(basis[i][j][2])
                    else:
                        base = base.at[i, j].set(basis[i][j][0])
            base = base.clip(-10, 10)
            diff = y - yhat
            quad = ((diff @ base) ** 2).sum()
            res = quad + alpha * ((diff ** 2).mean())
            return res
        
        def jnp_nocond_forward(yhat):
            dire = jnp.array((yhat > y), dtype=int)
            dire_col = jnp.expand_dims(dire, axis=1)
            dire_row = jnp.expand_dims(dire, axis=0)
            indices = jnp.expand_dims((dire_col + 2 * dire_row), axis=2)
            base = jnp.take_along_axis(basis, indices, axis=2).squeeze()
            base = jnp.tril(base)
            base = jnp.clip(base, -10, 10)
            diff = y - yhat
            quad = ((diff @ base) ** 2).sum()
            res = quad + alpha * ((diff ** 2).mean())
            return res
        if ret_naive:
            return jnpforward, jnp_nocond_forward
        import jax    
        return jax.jit(jnp_nocond_forward)



class LowRankQuadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self,
        Y,  # true labels
        rank=2,  # rank of the learned matrix
        quadalpha=0.1,  # regularisation weight
        **kwargs
    ):
        super(LowRankQuadratic, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))
        self.Y.requires_grad = False

        if "weights_vec" in kwargs:
            assert len(kwargs["weights_vec"].shape) == 2
            assert (kwargs["weights_vec"].shape[0] == self.Y.shape[0])
            assert (kwargs["weights_vec"].shape[1] == rank)
            basis = torch.tril(torch.tensor(kwargs["weights_vec"]) / (self.Y.shape[0] * self.Y.shape[0])).float()
        else:
            # Create a quadratic matrix
            basis = torch.tril(torch.rand((self.Y.shape[0], rank)) / (self.Y.shape[0] * self.Y.shape[0]))
        self.basis = torch.nn.Parameter(basis)  

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[:-len(self.Y_raw.shape)], self.Y.shape[0]))

        # Measure distance between predicted and true distributions
        diff = self.Y - Yhat

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = torch.tril(self.basis).clamp(-100, 100)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        # Add MSE as regularisation
        mse = diff.square().mean(dim=-1)

        return quad + self.alpha * mse

    def my_grad_hess(self, yhat, y):
        yhat = yhat.flatten()
        basis = torch.tril(self.basis).clamp(-100, 100).detach().cpu().numpy()
        y = y.flatten()
        diff = yhat - y
        hmat = basis @ basis.T
        hmat = hmat + hmat.T
        grad = (hmat @ diff) + 2 * self.alpha * diff/len(y)
        hess = np.diagonal(hmat) + 2 * self.alpha / len(y)

        return grad, hess

    def get_jnp_fun(self):
        import jax.numpy as jnp
        basis = jnp.array(torch.tril(self.basis).clamp(-100, 100).detach().cpu().numpy())
        alpha = self.alpha
        y = jnp.array(self.Y.detach().cpu().numpy())

        def jnpforward(yhat):
            diff = y - yhat
            quad = ((diff @ basis) ** 2).sum()
            mse = (diff ** 2).mean()
            res = quad + alpha * mse
            return res
        import jax
        return jax.jit(jnpforward)



## for debug
def test_model_jax(loss_model, Y=None, check_naive=False, **kwargs):
    if Y == None:
        Y = torch.rand((3, 2))
    Yhd = torch.rand_like(Y)
    yh = Yhd.detach().numpy()
    print(loss_model.__name__)

    import jax.numpy as jnp
    loss = loss_model(Y, **kwargs)
    print(loss(Yhd).item())
    if check_naive:
        fn, fn2 = loss.get_jnp_fun(True)
    else: 
        fn =  loss.get_jnp_fun()
    print(fn(jnp.array(yh.flatten())))

    mygrad, myhess = loss.my_grad_hess(yh, Y.detach().numpy())
    from jax import grad, jacfwd
    g = grad(fn)(jnp.array(yh.flatten()))
    jh = jnp.diagonal(jacfwd(grad(fn))(jnp.array(yh.flatten())))
    diff = ((g - mygrad) ** 2).sum()
    hdiff = ((jh - myhess) ** 2).sum()
    if not np.isclose(diff, 0):
        print("error grad diff {}".format(diff))
    if not np.isclose(hdiff, 0):
        print("error hess diff {}".format(hdiff))
    print("mygrad ", mygrad)
    print("jax g ", g)
    print("myhess ", myhess)
    print("jax h ", jh)

    if check_naive:
        g2 = grad(fn2)(jnp.array(yh.flatten()))
        jh2 = jnp.diagonal(jacfwd(grad(fn2))(jnp.array(yh.flatten())))
        diff = ((g2 - mygrad) ** 2).sum()
        hdiff = ((jh2 - myhess) ** 2).sum()
        if not np.isclose(diff, 0):
            print("error grad {}".format(diff))
        if not np.isclose(hdiff, 0):
            print("error hess {}".format(hdiff))
        print("mygrad ", mygrad)
        print("jax g2 ", g2)
        print("myhess ", myhess)
        print("jax h2 ", jh2)

    #import pdb
    #pdb.set_trace()


def test_jax():
    import jax.numpy as jnp
    from jax import grad
    weights = jnp.array(np.random.rand(10))
    y = jnp.array(np.random.rand(10))
    def mseloss(yhat):
        diff = (yhat - y)
        return (weights * (diff ** 2)).sum()

    print(grad(mseloss)(y + 0.1))
    print(2 * weights * 0.1)



def test():
    test_jax()
    test_model_jax(WeightedCE)
    test_model_jax(WeightedCE, torch.rand(8, 7))
    test_model_jax(WeightedMSE)
    test_model_jax(WeightedMSE, torch.rand(3, 5), weights_vec=np.random.rand(15))
    test_model_jax(WeightedMSE, torch.rand(3, 5), weights_vec=np.ones(15) * 100)
    test_model_jax(WeightedMSESum, torch.rand(5, 3))
    test_model_jax(WeightedMSESum, torch.rand(5, 2))
    test_model_jax(WeightedMSESum, torch.rand(3, 2))
    test_model_jax(WeightedMSESum, torch.rand(8, 3))
    test_model_jax(WeightedMSEPlusPlus)
    test_model_jax(LowRankQuadratic)
    test_model_jax(LowRankQuadratic, torch.rand(50, 70))
    test_model_jax(QuadraticPlusPlus)
    test_model_jax(QuadraticPlusPlus, check_naive=True)
    test_model_jax(QuadraticPlusPlus, torch.rand(5, 7), check_naive=True)
    test_model_jax(QuadraticPlusPlus, torch.rand(50, 70))


if __name__ == "__main__":
    test()

model_dict = {"dense": dense_nn, "dense_coupled": NNCoupled}
