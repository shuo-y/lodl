import random
import torch
import pdb
import os
import time
import pickle
# import matplotlib.pyplot as plt
from torch.multiprocessing import Pool
from statistics import mean
from functools import partial
from copy import deepcopy
import numpy as np
import xgboost as xgb

from models import DenseLoss, LowRankQuadratic, WeightedMSESum, WeightedMSE, WeightedCE, WeightedMSESum, QuadraticPlusPlus, WeightedMSEPlusPlus
from BudgetAllocation import BudgetAllocation
from BipartiteMatching import BipartiteMatching
from RMAB import RMAB
from utils import starmap_with_kwargs
NUM_CPUS = os.cpu_count()

# An default way from https://stackoverflow.com/questions/71178313/xgboost-custom-squarederror-loss-function-not-working-similar-to-default-impleme
def squared_error(predt: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label().reshape(predt.shape)
    grad = 0.1 * (y - predt)
    hess = 0.01 * np.ones(y.shape)
    return grad.flatten(), hess.flatten()

class search_weights_loss():
    def __init__(self, weights_vec, verbose=True):
        self.ypred_dim = len(weights_vec)
        assert len(weights_vec) == self.ypred_dim
        self.weights_vec = weights_vec
        self.logger = []


    def get_obj_fn(self):
        def grad_fn(predt, dtrain):
            y = dtrain.get_label().reshape(predt.shape)

            diff = (predt - y) #/ self.ypred_dim
            grad = 2 * self.weights_vec * diff
            hess = (2 * self.weights_vec) #/ self.ypred_dim

            hess = np.tile(hess, predt.shape[0]).reshape(predt.shape[0], self.ypred_dim)
            grad = grad.reshape(y.size)
            hess = hess.reshape(y.size)
            self.logger.append([predt, grad, hess])
            return grad, hess
        return grad_fn

    def get_eval_fn(self):
        def eval_fn(predt, dtrain):
            y = dtrain.get_label().reshape(predt.shape)
            diff = self.weights_vec * ((predt - y) ** 2)
            loss = diff.mean()
            return "evalloss", loss
        return eval_fn

class search_full_weights():
    def __init__(self, weights_mat, **kwargs):
        # Make weights the same shape as y
        self.weights_mat = weights_mat

    def get_obj_fn(self):
        def grad_fn(predt, dtrain):
            y = dtrain.get_label().reshape(predt.shape)

            diff = (predt - y)
            grad = 2 * self.weights_mat * diff
            hess = 2 * self.weights_mat

            grad = grad.flatten()
            hess = hess.flatten()
            return grad, hess
        return grad_fn

    def get_eval_fn(self):
        def eval_fn(predt, dtrain):
            y = dtrain.get_label().reshape(predt.shape)

            diff = (predt - y)
            return (2 * self.weights_mat * (diff ** 2))
        return eval_fn

class search_weights_directed_loss():
    def __init__(self, weights_vec, **kwargs):
        self.ypred_dim = len(weights_vec) // 2

        self.weights_pos = weights_vec[:self.ypred_dim]
        self.weights_neg = weights_vec[self.ypred_dim:]

    def get_obj_fn(self):
        def grad_fn(predt, dtrain):
            y = dtrain.get_label().reshape(predt.shape)

            diff = (predt - y)
            posdiff = (diff >= 0) * diff
            negdiff = (diff < 0) * diff

            grad = (2 * (self.weights_pos * posdiff) + 2 * (self.weights_neg * negdiff))
            hess = (2 * (self.weights_pos * (diff >= 0)) + 2 * (self.weights_neg * (diff < 0)))

            grad = grad.flatten()
            grad = grad #/ len(grad)  # TODO check if normalized matters when the feature is unrelated?
            hess = hess.flatten()
            hess = hess #/ len(hess) 
            return grad, hess
        return grad_fn

    def get_eval_fn(self):
        def eval_fn(predt, dtrain):
            y = dtrain.get_label().reshape(predt.shape)
            y = y.flatten()

            cur_pos_w = (predt > y) * self.weights_pos
            cur_neg_w = (predt < y) * self.weights_neg
            w = cur_pos_w + cur_neg_w
            diff = (predt - y)
            loss = (w * (diff ** 2)).mean()
            return "evaldirectedmseloss", loss
        return eval_fn



class search_quadratic_loss():
    def __init__(self, basis, alpha, **kwargs):
        self.ypred_dim = len(basis)
        self.basis = np.tril(basis)
        self.alpha = alpha
        #self.logger = []

    def get_obj_fn(self):
        def grad_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)

            diff = predt - y
            base = self.basis
            hmat = (base @ base.T)
            hmat = hmat + hmat.T
            grad = (diff @ hmat) + 2 * self.alpha * (diff/y.shape[1])
            grad = grad.reshape(y.size)
            hess = np.diagonal(hmat) + (2 * self.alpha/y.shape[1])
            hess = np.tile(hess, y.shape[0])
            #print(grad.sum())
            #print(hess.sum())
            #self.logger.append([grad, hess])
            return grad, hess
        return grad_fn

    def get_eval_fn(self):
        def eval_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)
            diff = y - predt
            ## (100, 2)  (2)
            #print(diff.shape)
            #print(self.basis.shape)
            quad = ((diff @ self.basis) ** 2).sum()
            mse = (diff ** 2).mean()
            res = quad + self.alpha * mse
            return "quadloss4", res
        return eval_fn


# TODO class search_diadir_quadratic_loss():



def MSE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).square().mean()


def MAE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).abs().mean()


def CE(Yhats, Ys, **kwargs):
    return torch.nn.BCELoss()(Yhats.clip(0, 1.0).float(), Ys.clip(0, 1.0).float())

def MSE_Sum(
    Yhats,
    Ys,
    alpha=0.1,  # weight of MSE-based regularisation
    **kwargs
):
    """
    Custom loss function that the squared error of the _sum_
    along the last dimension plus some regularisation.
    Useful for the Submodular Optimisation problems in Wilder et. al.
    """
    # Check if prediction is a matrix/tensor
    assert len(Ys.shape) >= 2

    # Calculate loss
    sum_loss = (Yhats - Ys).sum(dim=-1).square().mean()
    loss_regularised = (1 - alpha) * sum_loss + alpha * MSE(Yhats, Ys)
    return loss_regularised

def _sample_points(
    Y,  # The set of true labels
    problem,  # The optimisation problem at hand
    sampling,  # The method for sampling points
    num_samples,  # Number of points with which to fit model
    Y_aux=None,  # Extra information needed to solve the problem
    sampling_std=None,  # Standard deviation for the training data
    num_restarts=10,  # The number of times to run the optimisation problem for Z_opt
    X=None,
    train_model=None,
):
    # Sample points in the neighbourhood
    #   Find the rough scale of the predictions
    try:
        Y_std = float(sampling_std)
    except TypeError:
        Y_std = torch.std(Y) + 1e-5
    #   Generate points
    if sampling == 'random':
        #   Create some noise
        Y_noise = torch.distributions.Normal(0, Y_std).sample((num_samples, *Y.shape))
        #   Add this noise to Y to get sampled points
        Yhats = (Y + Y_noise)
    elif sampling == 'random_uniform':
        #   Create some noise
        Y_noise = torch.distributions.Uniform(0, Y_std).sample((num_samples, *Y.shape))
        #   Add this noise to Y to get sampled points
        Yhats = (Y + Y_noise)
    elif sampling == 'random_dropout':
        #   Create some noise
        Y_noise = torch.distributions.Normal(0, Y_std).sample((num_samples, *Y.shape))
        #   Drop some of the entries randomly
        drop_idxs = torch.distributions.Bernoulli(probs=0.1).sample((num_samples, *Y.shape))
        Y_noise = Y_noise * drop_idxs
        #   Add this noise to Y to get sampled points
        Yhats = (Y + Y_noise)
    elif sampling == 'random_flip':
        assert 0 < Y_std < 1
        #   Randomly choose some indices to flip
        flip_idxs = torch.distributions.Bernoulli(probs=Y_std).sample((num_samples, *Y.shape))
        #   Flip chosen indices to get sampled points
        Yhats = torch.logical_xor(Y, flip_idxs).float()
    elif sampling == 'numerical_jacobian':
        #   Find some points using this
        Yhats_plus = Y + (Y_std * torch.eye(Y.numel())).view((-1, *Y.shape))
        Yhats_minus = Y - (Y_std * torch.eye(Y.numel())).view((-1, *Y.shape))
        Yhats = torch.cat((Yhats_plus, Yhats_minus), dim=0)
    elif sampling == 'random_jacobian':
        #   Find dimensions to perturb and how much to perturb them by
        idxs = torch.randint(Y.numel(), size=(num_samples,))
        idxs = torch.nn.functional.one_hot(idxs, num_classes=Y.numel())
        noise_scale = torch.distributions.Normal(0, Y_std).sample((num_samples,)).unsqueeze(dim=-1)
        noise = (idxs * noise_scale).view((num_samples, *Y.shape))
        #   Find some points using this
        Yhats = Y + noise
    elif sampling == 'random_hessian':
        #   Find dimensions to perturb and how much to perturb them by
        noise = torch.zeros((num_samples, *Y.shape))
        for _ in range(2):
            idxs = torch.randint(Y.numel(), size=(num_samples,))
            idxs = torch.nn.functional.one_hot(idxs, num_classes=Y.numel())
            noise_scale = torch.distributions.Normal(0, Y_std).sample((num_samples,)).unsqueeze(dim=-1)
            noise += (idxs * noise_scale).view((num_samples, *Y.shape))
        #   Find some points using this
        Yhats = Y + noise
    elif sampling == 'sample_iter' and X != None and train_model != None:
        ypred = train_model(X).detach().squeeze()
        Yhat = (ypred + Y) / 2
        noise = torch.zeros((num_samples, *Y.shape))
        for _ in range(2):
            idxs = torch.randint(Y.numel(), size=(num_samples,))
            idxs = torch.nn.functional.one_hot(idxs, num_classes=Y.numel())
            noise_scale = torch.distributions.Normal(0, Y_std).sample((num_samples,)).unsqueeze(dim=-1)
            noise += (idxs * noise_scale).view((num_samples, *Y.shape))
        Yhats = Yhat + noise
    else:
        raise LookupError()
    #   Make sure that the points are valid predictions
    if isinstance(problem, BudgetAllocation) or isinstance(problem, BipartiteMatching):
        Yhats = Yhats.clamp(min=0, max=1)  # Assuming Yhats must be in the range [0, 1]
    elif isinstance(problem, RMAB):
        Yhats /= Yhats.sum(-1, keepdim=True)

    # Calculate decision-focused loss for points
    opt = partial(problem.get_decision, isTrain=True, aux_data=Y_aux)
    obj = partial(problem.get_objective, aux_data=Y_aux)

    #   Calculate for 'true label'
    best = None
    assert num_restarts > 0
    for _ in range(num_restarts):
        Z_opt = opt(Y)
        opt_objective = obj(Y, Z_opt)

        if best is None or opt_objective > best[1]:
            best = (Z_opt, opt_objective)
    Z_opt, opt_objective = best

    #   Calculate for Yhats
    Zs = opt(Yhats, Z_init=Z_opt)
    objectives = obj(Y.unsqueeze(0).expand(*Yhats.shape), Zs)

    return (Y, opt_objective, Yhats, objectives)

def _learn_loss(
    problem,  # The problem domain
    dataset,  # The data set on which to train SL
    model_type,  # The model we're trying to fit
    num_iters=100,  # Number of iterations over which to train model
    lr=1,  # Learning rate with which to train the model
    verbose=False,  # print training loss?
    train_frac=0.3,  # fraction of samples to use for training
    val_frac=0.3,  # fraction of samples to use for testing
    val_freq=1,  # the number of training steps after which to check loss on val set
    print_freq=5,  # the number of val steps after which to print losses
    patience=25,  # number of iterations to wait for the train loss to improve when learning
    **kwargs
):
    """
    kwargs
    no_train=True, # If truly train it or just initialize it
    Function that learns a model to approximate the behaviour of the
    'decision-focused loss' from Wilder et. al. in the neighbourhood of Y
    """
    # Get samples from dataset
    Y, opt_objective, Yhats, objectives = dataset
    objectives = opt_objective - objectives

    # Split train and test
    assert train_frac + val_frac < 1
    train_idxs = range(0, int(train_frac * Yhats.shape[0]))
    val_idxs = range(int(train_frac * Yhats.shape[0]), int((train_frac + val_frac) * Yhats.shape[0]))
    test_idxs = range(int((train_frac + val_frac) * Yhats.shape[0]), Yhats.shape[0])

    Yhats_train, objectives_train = Yhats[train_idxs], objectives[train_idxs]
    Yhats_val, objectives_val = Yhats[val_idxs], objectives[val_idxs]
    Yhats_test, objectives_test = Yhats[test_idxs], objectives[test_idxs]

    # Load a model
    if model_type == 'dense':
        model = DenseLoss(Y)
    elif model_type == 'quad':
        model = LowRankQuadratic(Y, **kwargs)
    elif model_type == 'weightedmse':
        if 'no_train' in kwargs and kwargs['no_train'] == True:
            if 'weights_vec' in kwargs:
                model = WeightedMSE(Y, min_val=kwargs['input_args'].weights_min, magnify=kwargs['input_args'].mag_factor, weights_vec=kwargs['weights_vec'])
            else:
                model = WeightedMSE(Y, min_val=kwargs['input_args'].weights_min, magnify=kwargs['input_args'].mag_factor)
            # Get final loss on train samples
            pred_train = model(Yhats_train).flatten()
            train_loss = torch.nn.L1Loss()(pred_train, objectives_train).item()

            # Get loss on holdout samples
            pred_test = model(Yhats_test).flatten()
            loss = torch.nn.L1Loss()(pred_test, objectives_test)
            test_loss = loss.item()
            return model, train_loss, test_loss
        model = WeightedMSE(Y, min_val=kwargs['input_args'].weights_min, magnify=kwargs['input_args'].mag_factor)
    elif model_type == 'weightedmse++':
        model = WeightedMSEPlusPlus(Y)
    elif model_type == 'weightedce':
        model = WeightedCE(Y)
    elif model_type == 'weightedmsesum':
        model = WeightedMSESum(Y)
    elif model_type == 'quad++':
        model = QuadraticPlusPlus(Y, **kwargs)
    else:
        raise LookupError()

    # Use GPU if available
    # Comment out since some of the parameters in model does not require grad
    #if torch.cuda.is_available():
    #    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #    Yhats_train, Yhats_val, Yhats_test = Yhats_train.to(device), Yhats_val.to(device), Yhats_test.to(device)
    #    objectives_train, objectives_val, objectives_test = objectives_train.to(device), objectives_val.to(device), objectives_test.to(device)
    #    model = model.to(device)

    # Fit a model to the points
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = (float("inf"), None)
    time_since_best = 0
    for iter_idx in range(num_iters):
        # Define update step using "closure" function
        def loss_closure():
            optimizer.zero_grad()
            pred = model(Yhats_train).flatten()
            if not (pred >= -1e-3).all().item():
                print(f"WARNING: Prediction value < 0: {pred.min()}")
            loss = MSE(pred, objectives_train)
            loss.backward()
            return loss

        # Perform validation
        if iter_idx % val_freq == 0:
            # Get performance on val dataset
            pred_val = model(Yhats_val).flatten()
            loss_val = MSE(pred_val, objectives_val)

            # Print statistics
            if verbose and iter_idx % (val_freq * print_freq) == 0:
                print(f"Iter {iter_idx}, Train Loss MSE: {loss_closure().item()}")
                print(f"Iter {iter_idx}, Val Loss MSE: {loss_val.item()}")
            # Save model if it's the best one
            if best[1] is None or loss_val.item() < best[0]:
                best = (loss_val.item(), deepcopy(model))
                time_since_best = 0
            # Stop if model hasn't improved for patience steps
            if time_since_best > patience:
                break

        # Make an update step
        optimizer.step(loss_closure)
        time_since_best += 1
    model = best[1]
    # If needed, PSDify
    # TODO: Figure out a better way to do this?
    # if hasattr(model, 'PSDify') and callable(model.PSDify):
    #     model.PSDify()

    # Get final loss on train samples
    pred_train = model(Yhats_train).flatten()
    train_loss = torch.nn.L1Loss()(pred_train, objectives_train).item()

    # Get loss on holdout samples
    pred_test = model(Yhats_test).flatten()
    loss = torch.nn.L1Loss()(pred_test, objectives_test)
    test_loss = loss.item()

    # Visualise generated datapoints and model
    # #   Visualise results on sampled_points
    # Yhats_flat = Yhats_train.reshape((Yhats_train.shape[0], -1))
    # Y_flat = Y.flatten()
    # Y_idx = random.randrange(Yhats_flat.shape[-1])
    # sample_idx = (Yhats_flat[:, Y_idx] - Y_flat[Y_idx]).square() > 0
    # pred = model(Yhats_train)
    # plt.scatter((Yhats_flat - Y_flat)[sample_idx, Y_idx].tolist(), objectives_train[sample_idx].tolist(), label='true')
    # plt.scatter((Yhats_flat - Y_flat)[sample_idx, Y_idx].tolist(), pred[sample_idx].tolist(), label='pred')
    # plt.legend(loc='upper right')
    # plt.show()

    # # Visualise results on random_direction
    # Y_flat = Y.flatten()
    # direction = 2 * torch.rand_like(Y_flat) - 1
    # direction = direction / direction.norm()
    # Y_range = 1
    # scale = torch.linspace(-Y_range, Y_range, 1000)
    # Yhats_flat = scale.unsqueeze(-1) * direction.unsqueeze(0) + Y_flat.unsqueeze(0)
    # pred = model(Yhats_flat)
    # true_dl = problem.get_objective(problem.get_decision(Yhats_flat), Yhats_flat) - problem.get_objective(problem.get_decision(Y_flat), Y_flat)
    # plt.scatter(scale.tolist(), true_dl.tolist(), label='true')
    # plt.scatter(scale.tolist(), pred.tolist(), label='pred')
    # plt.legend(loc='upper right')
    # plt.show()

    #if torch.cuda.is_available():
    #    model = model.to("cpu")

    return model, train_loss, test_loss


def _get_learned_loss(
    problem,
    model_type='weightedmse',
    folder='models',
    num_samples=400,
    sampling='random',
    sampling_std=None,
    serial=True,
    get_loss_model=False,
    samples_filename_read='',
    **kwargs
):
    print("Learning Loss Functions...")

    # Learn Losses
    #   Get Ys
    _, Y_train, Y_train_aux = problem.get_train_data()
    _, Y_val, Y_val_aux = problem.get_val_data()

    #   Get points in the neighbourhood of the Ys
    #       Try to load sampled points
    #  If error here check if the problem domain is VMScheduling
    if len(samples_filename_read) == 0:
        samples_filename_read = f"{problem.__class__.__name__}_{sampling}_{sampling_std}.pkl"


    samples_filename_write = f"{samples_filename_read[:-4]}_{time.time()}.pkl"

    # Check if there are enough stored samples
    num_samples_needed = num_extra_samples = num_samples
    if os.path.exists(samples_filename_read):
        with open(samples_filename_read, 'rb') as filehandle:
            print(f"read sample data file {samples_filename_read}")
            num_existing_samples, SL_dataset_old = pickle.load(filehandle)
    else:
        num_existing_samples = 0
        SL_dataset_old = {partition: [(Y, None, None, None) for Y in Ys] for Ys, partition in zip([Y_train, Y_val], ['train', 'val'])}

    # Sample more points if needed
    num_samples_needed = num_samples
    num_extra_samples = max(num_samples_needed - num_existing_samples, 0)
    datasets = [entry for entry in zip([Y_train, Y_val], [Y_train_aux, Y_val_aux], ['train', 'val'])]
    if num_extra_samples > 0:
        SL_dataset = {partition: [(Y, None, None, None) for Y in Ys] for Ys, partition in zip([Y_train, Y_val], ['train', 'val'])}
        for Ys, Ys_aux, partition in datasets:
            # Get new sampled points
            start_time = time.time()
            if serial == True:
                sampled_points = [_sample_points(Y, problem, sampling, num_extra_samples, Y_aux, sampling_std) for Y, Y_aux in zip(Ys, Ys_aux)]
            else:
                with Pool(NUM_CPUS) as pool:
                    sampled_points = pool.starmap(_sample_points, [(Y, problem, sampling, num_extra_samples, Y_aux, sampling_std) for Y, Y_aux in zip(Ys, Ys_aux)])
            print(f"({partition}) Time taken to generate {num_extra_samples} samples for {len(Ys)} instances: {time.time() - start_time}")

            # Use them to augment existing sampled points
            for idx, (Y, opt_objective, Yhats, objectives) in enumerate(sampled_points):
                SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)

        if "input_args" in kwargs and kwargs["input_args"].dumpsample == True:
            with open(samples_filename_write, 'wb') as filehandle:
                print(f"write sample files {samples_filename_write}")
                pickle.dump((num_extra_samples, SL_dataset), filehandle)

        #   Augment with new data
        for Ys, Ys_aux, partition in datasets:
            for idx, Y in enumerate(Ys):
                # Get old samples
                Y_old, opt_objective_old, Yhats_old, objectives_old = SL_dataset_old[partition][idx]
                Y_new, opt_objective_new, Yhats_new, objectives_new = SL_dataset[partition][idx]
                assert torch.isclose(Y_old, Y).all()
                assert torch.isclose(Y_new, Y).all()

                # Combine entries
                opt_objective = opt_objective_new if opt_objective_old is None else max(opt_objective_new, opt_objective_old)
                Yhats = Yhats_new if Yhats_old is None else torch.cat((Yhats_old, Yhats_new), dim=0)
                objectives = objectives_new if objectives_old is None else torch.cat((objectives_old, objectives_new), dim=0)

                # Update
                SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)
        num_existing_samples += num_extra_samples
    else:
        SL_dataset = SL_dataset_old

    #   Learn SL based on the sampled Yhats
    train_maes, test_maes, avg_dls = [], [], []
    losses = {}
    for Ys, Ys_aux, partition in datasets:
        # Sanity check that the saved data is the same as the problem's data
        for idx, (Y, Y_aux) in enumerate(zip(Ys, Ys_aux)):
            Y_dataset, opt_objective, _, objectives = SL_dataset[partition][idx]
            assert torch.isclose(Y, Y_dataset).all()

            # Also log the "average error"
            avg_dls.append((opt_objective - objectives).abs().mean().item())

        # Get num_samples_needed points
        random.seed(0)  # TODO: Remove. Temporary hack for reproducibility.
        idxs = random.sample(range(num_existing_samples), num_samples_needed)
        random.seed()

        # Learn a loss
        start_time = time.time()
        if serial == True:
            losses_and_stats = [_learn_loss(problem, (Y_dataset, opt_objective, Yhats[idxs], objectives[idxs]), model_type, **kwargs) for Y_dataset, opt_objective, Yhats, objectives in SL_dataset[partition]]
        else:
            with Pool(NUM_CPUS) as pool:
                losses_and_stats = starmap_with_kwargs(pool, _learn_loss, [(problem, (Y_dataset, opt_objective.detach().clone(), Yhats[idxs].detach().clone(), objectives[idxs].detach().clone()), deepcopy(model_type)) for Y_dataset, opt_objective, Yhats, objectives in SL_dataset[partition]], kwargs=kwargs)
        print(f"({partition}) Time taken to learn loss for {len(Ys)} instances: {time.time() - start_time}")

        # Parse and log results
        losses[partition] = []
        for learned_loss, train_mae, test_mae in losses_and_stats:
            train_maes.append(train_mae)
            test_maes.append(test_mae)
            losses[partition].append(learned_loss)

    # Print overall statistics
    print(f"\nMean Train DL - OPT: {mean(avg_dls)}")
    print(f"Train MAE for SL: {mean(train_maes)}")
    print(f"Test MAE for SL: {mean(test_maes)}\n")

    # Return the loss function in the expected form
    def surrogate_decision_quality(Yhats, Ys, partition, index, **kwargs):
        if partition == "test":
            print("This is a trained loss so it don't have value for the test set")
            return None
        return losses[partition][index](Yhats).flatten() - SL_dataset[partition][index][1]


    def jax_grad_hess(yhatnp, ynp, partition, index, **kwargs):
        from jax import grad, jacfwd
        import jax.numpy as jnp
        import numpy as np
        # The yhatnap and ynp should be np
        fn = losses[partition][index].get_jnp_fun()
        yinp = jnp.array(yhatnp.flatten())
        g = grad(fn)(yinp)
        h = jnp.diagonal(jacfwd(grad(fn))(yinp))
        g = np.array(g)
        h = np.array(h)
        return g, h

    def my_loss_model(yhatnp, ynp, partition, index, **kwargs):
        # yhatnp = yhatnp.flatten()
        # The yhatnap and ynp should be np
        return losses[partition][index] #.my_grad_hess(yhatnp)

    if (get_loss_model == True):
        return surrogate_decision_quality, my_loss_model
    return surrogate_decision_quality



def _get_decision_focused(
    problem,
    dflalpha=1.,
    **kwargs,
):
    if problem.get_twostageloss() == 'mse':
        twostageloss = MSE
    elif problem.get_twostageloss() == 'ce':
        twostageloss = CE
    else:
        raise ValueError(f"Not a valid 2-stage loss: {problem.get_twostageloss()}")

    def decision_focused_loss(Yhats, Ys, **kwargs):
        Zs = problem.get_decision(Yhats, isTrain=True, **kwargs)
        obj = problem.get_objective(Ys, Zs, isTrain=True, **kwargs)
        loss = -obj + dflalpha * twostageloss(Yhats, Ys)

        return loss

    return decision_focused_loss


def get_loss_fn(
    name,
    problem,
    **kwargs
):
    if name == 'mse':
        return MSE
    elif name == 'msesum':
        return MSE_Sum
    elif name == 'ce':
        return CE
    elif name == 'dfl':
        return _get_decision_focused(problem, **kwargs)
    else:
        return _get_learned_loss(problem, name, **kwargs)

if __name__ == "__main__":
    ytrain = np.random.random((10, 2))
    xtrain = np.random.random((10, 5))
    ytest = np.random.random((10, 2))
    weightloss = search_weights_loss(ytrain.shape[1], np.array([1.0 for _ in range(ytrain.shape[1])]))
    losfnweight = weightloss.get_obj_fn()
    dirweightloss = search_weights_directed_loss(ytrain.shape[1], np.array([1.0 for _ in range(ytrain.shape[1])]))
    losfndirweight = dirweightloss.get_obj_fn()

    Xy = xgb.DMatrix(xtrain, ytrain)
    grad, hess = losfnweight(ytest, Xy)
    graddir, hessdir = losfndirweight(ytest, Xy)


