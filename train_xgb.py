import xgboost as xgb
import torch
import numpy as np

class custom_tree:
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

def train_xgb(args, problem):
    # 2stage xgboost decoupled version
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    print("X_train.shape {}".format(X_train.shape))
    print("Y_train.shape {}".format(Y_train.shape))

    ## Squeeze to the last feature
    Xtrain = X_train.numpy().reshape(-1, X_train.shape[-1])
    Ytrain = Y_train.numpy().reshape(-1, Y_train.shape[-1])

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

class treefromlodl:
    # This class to be called by torch data type
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
        #g = np.zeros(self.ygold.shape).reshape(self.ygold.shape[0], np.prod(self.ygold.shape[1:]))
        #h = np.zeros(self.ygold.shape).reshape(self.ygold.shape[0], np.prod(self.ygold.shape[1:]))
        #for i in range(self.ygold.shape[0]):
        #    g[i], h[i] = self.grad_hess_fn(self.ygold[i], self.ygold[i], "train", i)
        #import pdb
        #pdb.set_trace()
        #print("check g sum {}".format(g.sum()))
        #print("check h sum {}".format(h.sum()))

        #print("initial custom_loss")
        #print(ygold[0])

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

            """
            print("grad() {}".format(grad))
            print("hes() {}".format(hes))
            y = dtrain.get_label().reshape(predt.shape)
            print(f"predt.size {predt.size}")
            check_grad = 2 * (predt - y).reshape(y.size)
            print(f"grad should be {check_grad}")
            if (all(np.isclose(check_grad, grad)) == False):
                print("error not close")

            y = dtrain.get_label().reshape(predt.shape)
            
            manual_grad = (predt - y).reshape(y.size)
            if np.isclose(grad, manual_grad).all():
                print("pass test")
            else:
                print("not pass test")
            print(manual_grad[:10])
            print(grad[:10]) 
            print(manual_grad[-10:])
            print(grad[-10:])
            """
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




def train_xgb_lodl(args, problem):
    from losses import get_loss_fn
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    Xtrain = X_train.numpy().reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    Ytrain = Y_train.numpy().reshape(Y_train.shape[0], np.prod(Y_train.shape[1:]))


    Xval = X_val.numpy().reshape(X_val.shape[0], np.prod(X_val.shape[1:]))
    Yval = Y_val.numpy().reshape(Y_val.shape[0], np.prod(Y_val.shape[1:]))

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
        model = xgbwrapper(reg, Y_train[0].shape)
        metrics = print_metrics(model, problem, args.loss, get_loss_fn(args.evalloss, problem), "seed{}".format(args.seed), isTrain=False)
        return model, metrics
    elif args.model == "xgb_coupled_clf":
        print("Use xgb.XGBClassifier")
        print("Use ce loss no matter the input args.loss")
        clf = xgb.XGBClassifier(tree_method=args.tree_method, n_estimators=args.num_estimators, learning_rate=args.tree_eta,
                               reg_alpha=args.tree_alpha, reg_lambda=args.tree_lambda)
        clf.fit(Xtrain, Ytrain, eval_set=[(Xtrain, Ytrain)])
        model = xgbwrapper(clf, Y_train[0].shape)
        if args.dumptree:
            dump_booster(clf.get_booster(), args)
        metrics = print_metrics(model, problem, args.loss, get_loss_fn(args.evalloss, problem), "seed{}".format(args.seed), isTrain=False)
        return model, metrics

    if args.weights_vec != '':
        weights_vec = eval(args.weights_vec)
    else:
        weights_vec = []

    print(f"Loading {args.loss} Loss Function...")
    loss_fn, loss_model_fn = get_loss_fn(
        args.loss,
        problem,
        sampling=args.sampling,
        num_samples=args.numsamples,
        rank=args.quadrank,
        sampling_std=args.samplingstd,
        quadalpha=args.quadalpha,
        lr=args.losslr,
        serial=args.serial,
        dflalpha=args.dflalpha,
        verbose=args.lodlverbose,
        get_loss_model=True,
        samples_filename_read=args.samples_read,
        no_train=args.no_train,
        weights_vec=weights_vec,
        input_args=args
    )

    # Based on some code from https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html

    cusloss = custom_loss(Y_train.detach().numpy(), loss_model_fn, loss_fn, args.mag_factor)
    obj_fun = cusloss.get_obj_fn()
    eval_fun = cusloss.get_eval_fn()

    #reg = xgb.XGBRegressor(tree_method='hist', n_estimators=args.num_estimators)
    Xy = xgb.DMatrix(Xtrain, Ytrain)
    Xyval = xgb.DMatrix(Xval, Yval)

    booster = xgb.train({"tree_method": args.tree_method, "num_target": Ytrain.shape[1], "lambda": args.tree_lambda,
                         "alpha": args.tree_alpha, "eta": args.tree_eta},
                        dtrain=Xy,
                        num_boost_round = args.num_estimators,
                        obj = obj_fun,
                        evals = [(Xy, "train")],
                        custom_metric = eval_fun)

    if args.dumptree:
        dump_booster(booster, args)
    model = treefromlodl(booster, Y_train[0].shape)

    metrics = print_metrics(model, problem, args.loss, get_loss_fn(args.evalloss, problem), "seed{}".format(args.seed), isTrain=False)
    return model, metrics


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


    def get_obj_fn(self):
        def grad_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)

            diff = (predt - y) / self.ypred_dim
            grad = 2 * self.weights_vec * diff
            hess = (2 * self.weights_vec) / self.ypred_dim
            hess = np.tile(hess, self.num_item).reshape(self.num_item, self.ypred_dim)
            return grad.reshape(y.size), hess.reshape(y.size)
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

    def get_obj_fn(self):
        def grad_fn(predt: np.ndarray, dtrain: xgb.DMatrix):
            y = dtrain.get_label().reshape(predt.shape)

            diff = predt - y
            base = self.basis
            hmat = (base @ base.T)
            grad = (diff @ hmat) + 2 * self.alpha * (diff/y.shape[1])
            grad = grad.reshape(y.size)
            hess = np.diagonal(hmat) + (2 * self.alpha/y.shape[1])
            hess = np.tile(hess, self.num_item)
            #print(grad.sum())
            #print(hess.sum())
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



def train_xgb_search_weights(args, problem):
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    Xtrain = X_train.numpy().reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    Ytrain = Y_train.numpy().reshape(Y_train.shape[0], np.prod(Y_train.shape[1:]))


    Xval = X_val.numpy().reshape(X_val.shape[0], np.prod(X_val.shape[1:]))
    Yval = Y_val.numpy().reshape(Y_val.shape[0], np.prod(Y_val.shape[1:]))


    Nsamples = args.search_numsamples
    # Sample a lot of N
    Nsub = args.search_subsamples
    # Select the sub part
    ndim = Ytrain.shape[1] * args.search_rank
    means = np.ones(ndim) * args.mag_factor
    covs = np.eye(ndim)
    print(f"The dimensiona is {ndim}")
    global_step = 0

    from losses import get_loss_fn

    for it in range(args.iters):
        obj_list = []
        weight_samples = np.random.multivariate_normal(means, covs, Nsamples)
        print(f"Iter {it}: means {means}  covs {covs}")
        for cnt in range(Nsamples):
            if args.search_rank > 1:
                cusloss = search_quadratic_loss(Ytrain.shape[0], Ytrain.shape[1], np.ones((Ytrain.shape[1], args.search_rank)).reshape(Ytrain.shape[1], args.search_rank), args.quadalpha)
            else:
                cusloss = search_weights_loss(Ytrain.shape[0], Ytrain.shape[1], weight_samples[cnt])

            Xy = xgb.DMatrix(Xtrain, Ytrain)
            booster = xgb.train({"tree_method": args.tree_method, "num_target": Ytrain.shape[1],
                                 "lambda": args.tree_lambda, "alpha": args.tree_alpha, "eta": args.tree_eta},
                                dtrain = Xy,
                                num_boost_round = args.search_estimators,
                                obj = cusloss.get_obj_fn(),
                                evals = [(Xy, "train")],
                                custom_metric = cusloss.get_eval_fn())

            model = treefromlodl(booster, Y_train[0].shape)
            pred = model(X_train).squeeze()
            Zs_pred = problem.get_decision(pred, aux_data=Y_train_aux, isTrain=True)
            objectives = problem.get_objective(Y_train, Zs_pred, aux_data=Y_train_aux)
            objective = objectives.mean().item() * (-1)
            obj_list.append(objective)
        ## Sort obj updates means and covs
        obj_list = np.array(obj_list)
        inds = np.argsort(obj_list)
        sub_weights = weight_samples[inds[:Nsub]]

        means = np.mean(sub_weights, axis=0)
        covs = np.cov(sub_weights.T)

    weight_samples = np.random.multivariate_normal(means, covs, 1)

    if args.search_rank > 1:
        cusloss = search_quadratic_loss(Ytrain.shape[0], Ytrain.shape[1], weight_samples[0].reshape(Ytrain.shape[1], args.search_rank), args.quadalpha)
    else:
        cusloss = search_weights_loss(Ytrain.shape[0], Ytrain.shape[1], weight_samples[0])

    Xy = xgb.DMatrix(Xtrain, Ytrain)
    booster = xgb.train({"tree_method": args.tree_method, "num_target": Ytrain.shape[1],
                         "lambda": args.tree_lambda, "alpha": args.tree_alpha, "eta": args.tree_eta},
                        dtrain = Xy,
                        num_boost_round = args.num_estimators,
                        obj = cusloss.get_obj_fn(),
                        evals = [(Xy, "train")],
                        custom_metric = cusloss.get_eval_fn())

    model = treefromlodl(booster, Y_train[0].shape)

    from utils import print_metrics
    metrics = print_metrics(model, problem, args.loss, get_loss_fn(args.evalloss, problem), "seed{}".format(args.seed), isTrain=False)

    return model, metrics










