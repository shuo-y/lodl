import xgboost as xgb
import torch
import numpy as np

class custom_tree:
    # This one is for decoupled tree
    # First version flatten all dimensions of Y
    # Make instance callable based on https://medium.com/swlh/callables-in-python-how-to-make-custom-instance-objects-callable-too-516d6eaf0c8d
    def __init__(self, reg, num_inp, num_out, Yishape):
        self.reg = reg
        self.num_inp = num_inp
        self.num_out = num_out
        self.Yishape = Yishape

    def __call__(self, X: torch.Tensor):
        Xi = X.numpy().reshape(-1, X.shape[-1])
        Yi = self.reg.predict(Xi)
        Y = torch.tensor(Yi)

        # Used prod based on https://numpy.org/doc/stable/reference/generated/numpy.prod.html
        if np.prod(Xi.shape) > self.num_inp:
            # X is a batch
            Y = torch.reshape(Y, [X.shape[0]] + list(self.Yishape))
            return Y
        elif np.prod(Xi.shape) == self.num_inp:
            return Y
        else:
            # https://docs.python.org/3/library/exceptions.html
            raise ValueError

def train_xgb(args, problem, xtrain, ytrain):
    # 2stage xgboost decoupled version
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    print("X_train.shape {}".format(X_train.shape))
    print("Y_train.shape {}".format(Y_train.shape))

    ## Squeeze to the last feature
    Xtrain = X_train.numpy().reshape(-1, X_train.shape[-1])
    batch_sz = Xtrain.shape[0]
    assert batch_sz * np.prod(Y_train.shape) // batch_sz == np.prod(Y_train.shape)
    # If error check if the dimensions of the shapes is appropriate
    Ytrain = Y_train.numpy().reshape(batch_sz, np.prod(Y_train.shape) // batch_sz)
    print(f"Data shape used for XGB input {Xtrain.shape} output {Ytrain.shape}")

    Xval = X_val.numpy().reshape(-1, X_val.shape[-1])
    Yval = Y_val.numpy().reshape(-1, Y_val.shape[-1])

    reg = xgb.XGBRegressor(tree_method='hist', n_estimators=args.num_estimators)


    reg.fit(Xtrain, Ytrain, eval_set=[(Xtrain, Ytrain)])
    model = custom_tree(reg, np.prod(X_train[0].shape), np.prod(Y_train[0].shape), Y_train[0].shape)
    from utils import print_metrics
    from losses import get_loss_fn
    metrics = print_metrics(model, problem, args.loss, get_loss_fn('mse', problem), "seed{}".format(args.seed), isTrain=False)
    return model, metrics

#class ext_model:

class decoupledboosterwrapper:
    # This class to be called by torch data type
    # This one use inplace_predict which is used by booster
    def __init__(self, tree, Yishape):
        self.tree = tree
        self.Yishape = Yishape

    def __call__(self, X: torch.Tensor):
        xn = X.numpy().reshape(-1, X.shape[-1])
        yhatn = self.tree.inplace_predict(xn)
        Y = torch.tensor(yhatn)
        Y = torch.reshape(Y, [X.shape[0]] + list(self.Yishape))
        return Y

class treefromlodl:
    # This class to be called by torch data type
    # This one use inplace_predict which is used by booster
    def __init__(self, tree, Yishape):
        self.tree = tree
        self.Yishape = Yishape

    def __call__(self, X: torch.Tensor):
        xn = X.numpy().reshape(X.shape[0], np.prod(X.shape[1:]))
        yhatn = self.tree.inplace_predict(xn)
        Y = torch.tensor(yhatn)
        Y = torch.reshape(Y, [X.shape[0]] + list(self.Yishape))
        return Y

class xgbwrapper:
    # This class to be called by torch data type
    # This one use predict which is used by XGBRegressor
    def __init__(self, tree, Yishape):
        self.tree = tree
        self.Yishape = Yishape

    def __call__(self, X: torch.Tensor):
        xn = X.numpy().reshape(X.shape[0], np.prod(X.shape[1:]))
        yhatn = self.tree.predict(xn)
        Y = torch.tensor(yhatn)
        Y = torch.reshape(Y, [X.shape[0]] + list(self.Yishape))
        return Y

class custom_loss():
    def __init__(self, ygold, loss_model_fn, loss_fn, mag_factor, verbose=True):
        self.ygold = ygold
        self.loss_model_fn = loss_model_fn
        self.loss_fn = loss_fn
        self.mag_factor = mag_factor
        self.verbose = verbose
        self.logger = []


    # Need use static method ?
    # https://www.digitalocean.com/community/tutorials/python-static-method
    # The rmse loss is based on https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html
    def get_rmse_obj(self):
        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)
            return (predt - y).reshape(y.size), np.ones(predt.shape).reshape(predt.size)
        return obj

    def get_rmse_eval(self):
        def eval_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)
            return "RMSE", np.sqrt(np.sum((y - predt) ** 2))
        return eval_fn  
  
    def get_obj_fn(self):
        
        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            predt = predt.reshape(self.ygold.shape)
            grad = np.zeros(self.ygold.shape).reshape(self.ygold.shape[0], np.prod(self.ygold.shape[1:]))
            hes = np.zeros(self.ygold.shape).reshape(self.ygold.shape[0], np.prod(self.ygold.shape[1:]))
            
            
            for i in range(self.ygold.shape[0]):
                 lo_model = self.loss_model_fn(predt[i], self.ygold[i].flatten(), "train", i)
                 grad[i], hes[i] = lo_model.my_grad_hess(predt[i], self.ygold[i].flatten())

            grad = grad.flatten()
            hes = hes.flatten()
            if self.verbose:
                print("grad.sum() {}".format(grad.sum()))
                print("hes.sum() {}".format(hes.sum()))

            self.logger.append([predt, grad, hes])
            return grad, hes
        return obj

    def get_eval_fn(self):
        def eval_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            predt = predt.reshape(self.ygold.shape)
            Yhats = torch.tensor(predt)
            # Borrow from utils.py
            losses = []
            for i, Yh in enumerate(Yhats):
                losses.append(self.loss_fn(Yh, None, "train", i))
            losses = torch.stack(losses).flatten()
            loss = losses.mean().item()
            return "LODLloss", loss
        return eval_fn




def train_xgb_lodl(args, problem, xtrain, ytrain, **kwargs):
    from losses import get_loss_fn

    Xtrain = xtrain.reshape(xtrain.shape[0], np.prod(xtrain.shape[1:]))
    Ytrain = ytrain.reshape(ytrain.shape[0], np.prod(ytrain.shape[1:]))

    print(f"Data shape used for XGB input {Xtrain.shape} output {Ytrain.shape}")

    from utils import print_metrics
    if args.model == "xgb_coupled":
        # 2stage xgboost coupled version
        print("Using native xgb.XGBRegressor")
        print("This option will ignore the args.loss")
        reg = xgb.XGBRegressor(tree_method='hist', n_estimators=args.num_estimators, learning_rate=args.tree_eta,
                               reg_alpha=args.tree_alpha, reg_lambda=args.tree_lambda)
        reg.fit(Xtrain, Ytrain, eval_set=[(Xtrain, Ytrain)])
        if args.dumptree:
            dump_booster(reg.get_booster(), args)
        return reg.get_booster()

    print(f"Loading {args.loss} Loss Function...")
    # See https://stackoverflow.com/questions/27892356/add-a-parameter-into-kwargs-during-function-call also for appending an argument
    # TODO how to learn loss based on passing arguments
    loss_kwargs = {"sampling": args.sampling,
                   "num_samples": args.numsamples,
                   "rank": args.quadrank,
                   "sampling_std" : args.samplingstd,
                   "quadalpha": args.quadalpha,
                   "lr": args.losslr,
                   "serial": args.serial,
                   "dflalpha": args.dflalpha,
                   "verbose": args.verbose,
                   "get_loss_model": True,
                   "samples_filename_read": args.samples_read,
                   "input_args": args}
    if args.weights_vec != "":
        weights_vec = eval(args.weights_vec)
        loss_kwargs["weights_vec"] = weights_vec

    loss_fn, loss_model_fn = get_loss_fn(
        args.loss,
        problem,
        **loss_kwargs
    )

    # Based on some code from https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html

    cusloss = custom_loss(ytrain, loss_model_fn, loss_fn, args.mag_factor)
    obj_fun = cusloss.get_obj_fn()
    eval_fun = cusloss.get_eval_fn()

    Xy = xgb.DMatrix(Xtrain, Ytrain)

    booster = xgb.train({"tree_method": args.tree_method, "num_target": Ytrain.shape[1],
                         "lambda": args.tree_lambda, "alpha": args.tree_alpha, "eta": args.tree_eta,
                         "gamma": args.tree_gamma, "max_depth": args.tree_max_depth,
                         "min_child_weight": args.tree_min_child_weight,
                         "max_delta_step": args.tree_max_delta_step, "subsample": args.tree_subsample,
                         "scale_pos_weight": args.tree_scale_pos_weight},
                        dtrain=Xy,
                        num_boost_round = args.num_estimators,
                        obj = obj_fun,
                        evals = [(Xy, "train")],
                        custom_metric = eval_fun)

    if args.dumptree:
        dump_booster(booster, args)
    return booster


def dump_booster(booster, args):
    # Get time based on https://www.programiz.com/python-programming/datetime/current-time
    from datetime import datetime
    timestr = "models/" + args.model + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    import sys
    configfilename = timestr + ".config"
    treefilename = timestr + ".tree"
    with open(timestr + ".args", "w") as f:
        f.write(str(args))
        f.write("\n")
        f.write(configfilename)
        f.write("\n")
        f.write(treefilename)

    treedump = booster.get_dump()

    with open(treefilename, "w") as f:
        for cnt, tr in enumerate(treedump):
            f.write("=======tree{}=======\n".format(cnt))
            f.write(tr)

    config = booster.save_config()
    with open(configfilename, "w") as f:
        f.write(config)


class search_weights_loss():
    def __init__(self, num_item, ypred_dim, weights_vec, verbose=True):
        self.num_item = num_item
        self.ypred_dim = ypred_dim
        assert len(weights_vec) == self.ypred_dim
        self.weights_vec = weights_vec
        self.logger = []


    def get_obj_fn(self):
        def grad_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)

            diff = (predt - y) / self.ypred_dim
            grad = 2 * self.weights_vec * diff
            hess = (2 * self.weights_vec) / self.ypred_dim
            hess = np.tile(hess, self.num_item).reshape(self.num_item, self.ypred_dim)
            grad = grad.reshape(y.size)
            hess = hess.reshape(y.size)
            self.logger.append([predt, grad, hess])
            return grad, hess
        return grad_fn

    def get_eval_fn(self):
        def eval_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)
            diff = self.weights_vec * ((predt - y) ** 2)
            loss = diff.mean()
            return "evalloss", loss
        return eval_fn



class search_quadratic_loss():
    def __init__(self, num_item, ypred_dim, basis, alpha):
        self.num_item = num_item
        self.ypred_dim = ypred_dim
        self.basis = np.tril(basis)
        self.alpha = alpha
        self.logger = []

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
            hess = np.tile(hess, self.num_item)
            #print(grad.sum())
            #print(hess.sum())
            self.logger.append([predt, grad, hess])
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

class search_direct_quadratic_loss():
    def __init__(self, num_item, ypred_dim, basis, alpha):
        self.num_item = num_item
        self.ypred_dim = ypred_dim
        self.basis = np.tril(basis)
        assert len(self.basis.shape) == 3
        assert self.basis.shape[-1] == 4
        # basis should be ypred_dim * ypred_dim * 4
        self.logger = []
        self.alpha = alpha

    def get_obj_fn(self):
        def grad_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)

            grad = np.zeros(y.shape)
            hess = np.zeros(y.shape)

            for i in range(y.shape[0]):
                diff = predt[i] - y[i]
                base = self.basis

                direction = (predt[i] > y[i]).astype(int)
                direction_col = np.expand_dims(direction, axis=1)
                direction_row = np.expand_dims(direction, axis=0)
                indices = direction_col  + 2 * direction_row
                indices = np.expand_dims(indices, axis=2)

                base = np.squeeze(np.take_along_axis(self.basis, indices, axis=2))

                hmat = (base @ base.T)
                hmat = hmat + hmat.T
                grad[i] = (hmat @ diff) + 2 * self.alpha * (diff/y.shape[1])
                hess[i] = np.diagonal(hmat) + (2 * self.alpha/y.shape[1])

            grad = grad.flatten()
            hess = hess.flatten()

            self.logger.append([predt, grad, hess])

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
            return "directquadloss4", res
        return eval_fn


def cem_get_objective(problem, model, X, Y, Yaux):
    pred = model(X).squeeze()
    Zs_pred = problem.get_decision(pred, aux_data=Yaux, isTrain=True)
    objectives = problem.get_objective(Y, Zs_pred, aux_data=Yaux)
    return objectives.mean().item() * (-1)

def check_logger(logger, args):

    for i in range(0, len(logger) - 1):
        # Each entry in logger: predt, grad, hess
        ftx = ((logger[i + 1][0] - logger[i][0]).flatten()) / args.tree_eta   # where t = i
        step = - logger[i][1] / logger[i][2] #  - g_i / h_i
        if not all(np.isclose(ftx, step)):
            print("xgboost fit not close")
        meanabs = np.absolute(ftx - step).mean()
        relabs = meanabs / np.absolute(step).mean()
        print(f"At estimators {i} differences {meanabs} {relabs}")


def train_xgb_ngopt(args, problem):
    import nevergrad as ng
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()  # Xval is still needed for CEM

    Xtrain = X_train.numpy().reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    Ytrain = Y_train.numpy().reshape(Y_train.shape[0], np.prod(Y_train.shape[1:]))

    Xval = X_val.numpy().reshape(X_val.shape[0], np.prod(X_val.shape[1:]))
    Yval = Y_val.numpy().reshape(Y_val.shape[0], np.prod(Y_val.shape[1:]))

    Xy = xgb.DMatrix(Xtrain, Ytrain)

    def train_tree(weights_vec: np.ndarray) -> float:
        cusloss = search_weights_loss(Ytrain.shape[0], Ytrain.shape[1], weights_vec)
        booster = xgb.train({"tree_method": args.tree_method, "num_target": Ytrain.shape[1],
                                 "lambda": args.tree_lambda, "alpha": args.tree_alpha, "eta": args.tree_eta,
                                 "gamma": args.tree_gamma, "max_depth": args.tree_max_depth,
                                 "min_child_weight": args.tree_min_child_weight,
                                 "max_delta_step": args.tree_max_delta_step, "subsample": args.tree_subsample,
                                 "colsample_bytree": args.tree_colsample_bytree,
                                 "colsample_bylevel": args.tree_colsample_bylevel,
                                 "colsample_bynode": args.tree_colsample_bynode,
                                 "scale_pos_weight": args.tree_scale_pos_weight},
                                dtrain = Xy,
                                num_boost_round = args.search_estimators,
                                obj = cusloss.get_obj_fn())
        ypred = booster.inplace_predict(Xval)
        dloss = problem.dec_loss(ypred, Yval).mean()
        return dloss

    # Running in parallel https://facebookresearch.github.io/nevergrad/optimization.html#using-several-workers
    parametrization = ng.p.Instrumentation(weights_vec=ng.p.Array(shape=(Ytrain.shape[1],)))

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=args.ng_budget)
    recommendation = optimizer.minimize(train_tree)


    choose_weights = np.array(recommendation.value[1]['weights_vec'])
    print(f"Choose weights is {choose_weights}")

    cusloss = search_weights_loss(Ytrain.shape[0], Ytrain.shape[1], choose_weights)

    booster = xgb.train({"tree_method": args.tree_method, "num_target": Ytrain.shape[1],
                         "lambda": args.tree_lambda, "alpha": args.tree_alpha, "eta": args.tree_eta,
                         "gamma": args.tree_gamma, "max_depth": args.tree_max_depth,
                         "min_child_weight": args.tree_min_child_weight,
                         "max_delta_step": args.tree_max_delta_step, "subsample": args.tree_subsample,
                         "colsample_bytree": args.tree_colsample_bytree,
                         "colsample_bylevel": args.tree_colsample_bylevel,
                         "colsample_bynode": args.tree_colsample_bynode,
                         "scale_pos_weight": args.tree_scale_pos_weight},
                        dtrain = Xy,
                        num_boost_round = args.num_estimators,
                        obj = cusloss.get_obj_fn(),
                        evals = [(Xy, "train")],
                        custom_metric = cusloss.get_eval_fn())

    model = treefromlodl(booster, Y_train[0].shape)
    from losses import get_loss_fn
    from utils import print_metrics
    metrics = print_metrics(model, problem, args.loss, get_loss_fn("mse", problem), "seed{}".format(args.seed), isTrain=False)
    return model, metrics

def evaluate_one_search(params, problem, xtrain, ytrain, xval, yval, weight_vec):

    cusloss = search_weights_loss(ytrain.shape[0], ytrain.shape[1], params["mag_factor"] * weight_vec)

    Xy = xgb.DMatrix(xtrain, ytrain)

    booster = xgb.train({"tree_method": params["tree_method"],
                         "num_target": ytrain.shape[1],
                         "lambda": params["tree_lambda"],
                         "alpha": params["tree_alpha"],
                         "eta": params["tree_eta"],
                         "gamma": params["tree_gamma"],
                         "max_depth": params["tree_max_depth"],
                         "min_child_weight": params["tree_min_child_weight"],
                         "max_delta_step": params["tree_max_delta_step"],
                         "subsample": params["tree_subsample"],
                         "colsample_bytree": params["tree_colsample_bytree"],
                         "colsample_bylevel": params["tree_colsample_bylevel"],
                         "colsample_bynode": params["tree_colsample_bynode"],
                         "scale_pos_weight": params["tree_scale_pos_weight"]},
                        dtrain = Xy,
                        num_boost_round = params["search_estimators"],
                        obj = cusloss.get_obj_fn())

    ypred = booster.inplace_predict(xval)
    objective = problem.dec_loss(ypred, yval)
    objective = objective.squeeze().mean()
    return objective, booster

def cem_one_restart(params, problem, xtrain, ytrain, xval, yval):
    bestmodel_so_far = None
    bestweights_so_far = None
    best = None
    Nsamples = params["search_numsamples"]
    # Sample a lot of N
    Nsub = params["search_subsamples"]
    ndim = ytrain.shape[1]
    means = np.random.rand(ndim) * params["search_means"]
    randL = np.tril(np.random.rand(ndim, ndim) + 0.01)
    covs = randL @ randL.T
    covs = covs * params["search_covs"]

    for it in range(params["iters"]):
        weight_samples = np.random.multivariate_normal(means, covs, Nsamples)
        weight_samples = np.vstack((weight_samples, np.ones(ndim) * params["search_means"]))
        weight_samples = np.clip(weight_samples, 1.0, None)
        results = []

        for cnt in range(Nsamples + 1):
            #weight_sample = np.random.multivariate_normal(means, covs, 1)
            objective, model = evaluate_one_search(params, problem, xtrain, ytrain, xval, yval, weight_samples[cnt])
            results.append((objective, model))

        obj_list = np.array([obj for obj, _ in results])
        inds = np.argsort(obj_list)
        sub_weights = weight_samples[inds[:Nsub]]

        if best == None or obj_list[inds[0]] < best:
            best = obj_list[inds[0]]
            bestmodel_so_far = results[inds[0]][1]
            bestweights_so_far = weight_samples[inds[0]]
        means = np.mean(sub_weights, axis=0)
        covs = np.cov(sub_weights.T).reshape(ndim, ndim)

    return best, bestmodel_so_far, bestweights_so_far


def train_xgb_search_weights_multirestart(args, prob, probkwargs, xtrain, ytrain, xval, yval):
    problemcopies = [prob(**probkwargs) for _ in range(args.restart_rounds)]
    params = vars(args)
    import copy
    paramscopies = [copy.deepcopy(params) for _ in range(args.restart_rounds)]
    xtraincopies = [np.copy(xtrain) for _ in range(args.restart_rounds)]
    ytraincopies = [np.copy(ytrain) for _ in range(args.restart_rounds)]
    xvalcopies = [np.copy(xval) for _ in range(args.restart_rounds)]
    yvalcopies = [np.copy(yval) for _ in range(args.restart_rounds)]

    from torch.multiprocessing import Pool
    import os
    import time

    starttime = time.time()
    with Pool(os.cpu_count()) as pl:
        results = pl.starmap(cem_one_restart, [(paramscopies[cnt], problemcopies[cnt], xtraincopies[cnt], ytraincopies[cnt], xvalcopies[cnt], yvalcopies[cnt]) for cnt in range(args.restart_rounds)])

    print(f"Train time {time.time() - starttime}")

    obj_list = np.array([obj for obj, _ , _ in results])
    inds = np.argsort(obj_list)
    bestweights_so_far = results[inds[0]][2]

    print(f"Best weights so far {bestweights_so_far} sum{bestweights_so_far.sum()}")

    cusloss = search_weights_loss(ytrain.shape[0], ytrain.shape[1], args.mag_factor * bestweights_so_far)
    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train({"tree_method": args.tree_method, "num_target": ytrain.shape[1],
                         "lambda": args.tree_lambda, "alpha": args.tree_alpha, "eta": args.tree_eta,
                         "gamma": args.tree_gamma, "max_depth": args.tree_max_depth,
                         "min_child_weight": args.tree_min_child_weight,
                         "max_delta_step": args.tree_max_delta_step, "subsample": args.tree_subsample,
                         "colsample_bytree": args.tree_colsample_bytree,
                         "colsample_bylevel": args.tree_colsample_bylevel,
                         "colsample_bynode": args.tree_colsample_bynode,
                         "scale_pos_weight": args.tree_scale_pos_weight},
                        dtrain = Xy,
                        num_boost_round = args.num_estimators,
                        obj = cusloss.get_obj_fn(),
                        evals = [(Xy, "train")],
                        custom_metric = cusloss.get_eval_fn())
    return booster


def train_xgb_search_weights(args, prob, probkwargs, xtrain, ytrain, xval, yval):

    #if args.model == "xgb_search_decoupled":
    #    Xtrain = Xtrain.reshape(-1, X_train.shape[-1])
    #    batch_sz = Xtrain.shape[0]
    #    assert batch_sz * np.prod(Y_train.shape) // batch_sz == np.prod(Y_train.shape)
    #    # If error check if the dimensions of the shapes is appropriate
    #    Ytrain = Y_train.numpy().reshape(batch_sz, np.prod(Y_train.shape) // batch_sz)

    print(f"Data shape used for XGB input {xtrain.shape} output {ytrain.shape}")

    Nsamples = args.search_numsamples
    # Sample a lot of N
    Nsub = args.search_subsamples
    # Select the sub part
    ndim = ytrain.shape[1]

    print(f"The dimensiona is {ndim}")
    global_step = 0

    from losses import get_loss_fn
    from utils import print_metrics
    from torch.multiprocessing import Pool
    import os
    import time
    import copy

    outer_iter = args.restart_rounds
    bestmodel_so_far = None
    bestweights_so_far = np.ones(ndim) * args.search_means
    best = None

    params = vars(args)
    problem = prob(**probkwargs)
    paramscopies = [copy.deepcopy(params) for _ in range(Nsamples + 1)]
    problemcopies = [prob(**probkwargs) for _ in range(Nsamples + 1)]
    xtraincopies = [np.copy(xtrain) for _ in range(Nsamples + 1)]
    ytraincopies = [np.copy(ytrain) for _ in range(Nsamples + 1)]
    xvalcopies = [np.copy(xval) for _ in range(Nsamples + 1)]
    yvalcopies = [np.copy(yval) for _ in range(Nsamples + 1)]


    for oit in range(outer_iter):
        means = np.random.rand(ndim) * args.search_means
        randL = np.tril(np.random.rand(ndim, ndim) + 0.01)
        covs = randL @ randL.T
        covs = covs * params["search_covs"]

        for it in range(args.iters):
            weight_samples = np.random.multivariate_normal(means, covs, Nsamples)
            weight_samples = np.vstack((weight_samples, np.ones(ndim) * args.search_means))
            weight_samples = np.clip(weight_samples, 1.0, None)
            results = []

            if args.verbose:
                start_time = time.time()
                print(f"Iter {it}: means {means[:5]}...  covs {covs[:5]}...")
                if (oit * args.iters + it + 1) % 15 == 0:
                    yvalsum = yval.sum()
                    ytrueobj = problem.dec_loss(yval, yval)
                    print(f"Restart {oit} Iter {it} sanity check {yvalsum} {ytrueobj}")
                    #print(f"Weight vec{weight_samples}")
                    #print(f"Weight vec isna{np.isnan(weight_samples).any()}  max{np.max(weight_samples)}  min{np.min(weight_samples)}")

            if args.serial == True:
                for cnt in range(Nsamples + 1):
                    #weight_sample = np.random.multivariate_normal(means, covs, 1)
                    objective, model = evaluate_one_search(params, problem, xtrain, ytrain, xval, yval, weight_samples[cnt])
                    results.append((objective, model))

            else:
                # TODO multi threads has some issue currently
                with Pool(os.cpu_count()) as pl:
                    results = pl.starmap(evaluate_one_search, [(paramscopies[cnt], problemcopies[cnt], xtraincopies[cnt], ytraincopies[cnt], xvalcopies[cnt], yvalcopies[cnt], weight_samples[cnt]) for cnt in range(Nsamples + 1)])


            ## Sort obj updates means and covs
            obj_list = np.array([obj for obj, _ in results])
            inds = np.argsort(obj_list)
            sub_weights = weight_samples[inds[:Nsub]]

            if args.verbose:
                select_model = results[inds[0]][1]
                print(f"Iter {it} best obj is {obj_list[inds[0]]} Time of one iteration CEM is {time.time() - start_time}")

            if best == None or obj_list[inds[0]] < best:
                best = obj_list[inds[0]]
                bestmodel_so_far = results[inds[0]][1]
                bestweights_so_far = weight_samples[inds[0]]

            means = np.mean(sub_weights, axis=0)
            covs = np.cov(sub_weights.T).reshape(ndim, ndim)

        #weight_samples = np.random.multivariate_normal(means, covs, 1)
        #print(f"outer_iter{oit}: means {means} covs {covs} weights {weight_samples[0]}")

    print(f"Best weights so far {bestweights_so_far} sum{bestweights_so_far.sum()}")
    cusloss = search_weights_loss(ytrain.shape[0], ytrain.shape[1], args.mag_factor * bestweights_so_far)

    Xy = xgb.DMatrix(xtrain, ytrain)
    booster = xgb.train({"tree_method": args.tree_method, "num_target": ytrain.shape[1],
                         "lambda": args.tree_lambda, "alpha": args.tree_alpha, "eta": args.tree_eta,
                         "gamma": args.tree_gamma, "max_depth": args.tree_max_depth,
                         "min_child_weight": args.tree_min_child_weight,
                         "max_delta_step": args.tree_max_delta_step, "subsample": args.tree_subsample,
                         "colsample_bytree": args.tree_colsample_bytree,
                         "colsample_bylevel": args.tree_colsample_bylevel,
                         "colsample_bynode": args.tree_colsample_bynode,
                         "scale_pos_weight": args.tree_scale_pos_weight},
                        dtrain = Xy,
                        num_boost_round = args.num_estimators,
                        obj = cusloss.get_obj_fn(),
                        evals = [(Xy, "train")],
                        custom_metric = cusloss.get_eval_fn())


    if args.tree_check_logger:
        check_logger(cusloss.logger, args)

    return booster


def perf_booster(args, problem, booster, X, y, name):
    ypred = booster.inplace_predict(X)

    mseloss = ((y - ypred) ** 2).mean()

    randobjs = []
    for _ in range(10):
        yrandom = np.random.rand(*y.shape)
        randomobj = problem.dec_loss(yrandom, y)
        randobjs.append(randomobj)

    randobjs = np.array(randobjs)
    randobjs = randobjs.mean(axis = 0)
    predobj = problem.dec_loss(ypred, y)
    optobj = problem.dec_loss(y, y)

    randobjs = np.squeeze(randobjs)
    predobj = np.squeeze(predobj)
    optobj = np.squeeze(optobj)

    norobj = (predobj - randobjs) / (optobj - randobjs)
    norobj = norobj.mean()

    norobjnotperins = (predobj.mean() - randobjs.mean()) / (optobj.mean()- randobjs.mean())
    print(f"{name} Normalized DLoss (per instance) {norobj}  NorDLoss (not per instance) {norobjnotperins} DLoss {predobj.mean()} Random DLoss {randobjs.mean()} Opt DLOSS {optobj.mean()} MSE{mseloss}")
    csvstring = "%.12f,%.12f,%.12f,%.12f,%.12f,%.12f," % (norobj, norobjnotperins, predobj.mean(), randobjs.mean(), optobj.mean(), mseloss)
    return csvstring












