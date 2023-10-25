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

def train_xgb(args, problem):
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




def train_xgb_lodl(args, problem):
    from losses import get_loss_fn
    X_train, Y_train, Y_train_aux = problem.get_train_data()

    Xtrain = X_train.numpy().reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    Ytrain = Y_train.numpy().reshape(Y_train.shape[0], np.prod(Y_train.shape[1:]))

    if args.model == "xgb_lodl_decoupled":
        Xtrain = Xtrain.reshape(-1, X_train.shape[-1])
        batch_sz = Xtrain.shape[0]
        assert batch_sz * np.prod(Y_train.shape) // batch_sz == np.prod(Y_train.shape)
        # If error check if the dimensions of the shapes is appropriate
        Ytrain = Y_train.numpy().reshape(batch_sz, np.prod(Y_train.shape) // batch_sz)

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

    print(f"Loading {args.loss} Loss Function...")
    # See https://stackoverflow.com/questions/27892356/add-a-parameter-into-kwargs-during-function-call also for appending an argument
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

    cusloss = custom_loss(Y_train.detach().numpy(), loss_model_fn, loss_fn, args.mag_factor)
    obj_fun = cusloss.get_obj_fn()
    eval_fun = cusloss.get_eval_fn()

    #reg = xgb.XGBRegressor(tree_method='hist', n_estimators=args.num_estimators)
    print(f"Data shape used for XGB input {Xtrain.shape} output {Ytrain.shape}")
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
    model = treefromlodl(booster, Y_train[0].shape)
    if args.model == "xgb_lodl_decoupled":
        model = decoupledboosterwrapper(booster, Y_train[0].shape)

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

def evaluate_one_search(args, problem, Xtrain, Ytrain, weights_vec, ygoldshape):
    if args.loss == "weightedmse":
        cusloss = search_weights_loss(Ytrain.shape[0], Ytrain.shape[1], args.mag_factor * weights_vec)
    elif args.loss == "quad":
        cusloss = search_quadratic_loss(Ytrain.shape[0], Ytrain.shape[1], args.mag_factor * weights_vec.reshape(Ytrain.shape[1], args.quadrank), args.quadalpha)
    elif args.loss == "quad++":
        cusloss = search_direct_quadratic_loss(Ytrain.shape[0], Ytrain.shape[1], args.mag_factor * weights_vec.reshape(Ytrain.shape[1], Ytrain.shape[1], 4), args.quadalpha)

    Xy = xgb.DMatrix(Xtrain, Ytrain)

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
                        obj = cusloss.get_obj_fn(),
                        evals = eval(args.search_eval),
                        custom_metric = cusloss.get_eval_fn())

    if args.model == "xgb_search_decoupled":
        model = decoupledboosterwrapper(booster, ygoldshape)
    else:
        model = treefromlodl(booster, ygoldshape)

    X_val, Y_val, Y_val_aux = problem.get_val_data()
    objective = eval(args.search_obj)
    #if returnmodel == True:
    return objective, model
    #return objective


def train_xgb_search_weights(args, problem):
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()  # Xval is still needed for CEM

    Xtrain = X_train.numpy().reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    Ytrain = Y_train.numpy().reshape(Y_train.shape[0], np.prod(Y_train.shape[1:]))

    if args.model == "xgb_search_decoupled":
        Xtrain = Xtrain.reshape(-1, X_train.shape[-1])
        batch_sz = Xtrain.shape[0]
        assert batch_sz * np.prod(Y_train.shape) // batch_sz == np.prod(Y_train.shape)
        # If error check if the dimensions of the shapes is appropriate
        Ytrain = Y_train.numpy().reshape(batch_sz, np.prod(Y_train.shape) // batch_sz)

    print(f"Data shape used for XGB input {Xtrain.shape} output {Ytrain.shape}")

    Nsamples = args.search_numsamples
    # Sample a lot of N
    Nsub = args.search_subsamples
    # Select the sub part
    if args.loss == "weightedmse":
        ndim = Ytrain.shape[1]
    elif args.loss == "quad":
        ndim = Ytrain.shape[1] * args.quadrank
    elif args.loss == "quad++":
        ndim = Ytrain.shape[1] * Ytrain.shape[1] * 4
    else:
        assert "Not supported loss"

    means = np.ones(ndim) * args.search_means
    covs = np.eye(ndim) * args.search_covs
    print(f"The dimensiona is {ndim}")
    global_step = 0

    from losses import get_loss_fn
    from utils import print_metrics
    from torch.multiprocessing import Pool
    import os
    import time

    for it in range(args.iters):
        obj_list = []
        model_list = []
        if args.verbose:
            start_time = time.time()
            print(f"Iter {it}: means {means[:5]}...  covs {covs[:5]}...")

        weight_samples = np.random.multivariate_normal(means, covs, Nsamples)
        if args.serial == True:
            for cnt in range(Nsamples):
                objective, model = evaluate_one_search(args, problem, Xtrain, Ytrain, weight_samples[cnt], Y_train[0].shape)
                obj_list.append(objective)
                model_list.append(model)
        else:
            with Pool(os.cpu_count()) as p:
                results = p.starmap(evaluate_one_search, [(args, problem, Xtrain, Ytrain, weight_vec, Y_train[0].shape) for weight_vec in weight_samples])
                for obj, model in results:
                    obj_list.append(obj)
                    model_list.append(model)

        ## Sort obj updates means and covs
        obj_list = np.array(obj_list)
        inds = np.argsort(obj_list)
        sub_weights = weight_samples[inds[:Nsub]]

        if args.verbose:
            select_model = model_list[inds[0]]
            print_metrics(select_model, problem, args.loss, get_loss_fn(args.evalloss, problem), f"seed{args.seed}iter{it}best", isTrain=False)
            print(f"Time of one iteration CEM is {time.time() - start_time}")

        means = np.mean(sub_weights, axis=0)
        covs = np.cov(sub_weights.T).reshape(ndim, ndim)

    weight_samples = np.random.multivariate_normal(means, covs, 1)
    print(f"Final: means {means} covs {covs} weights {weight_samples[0]}")

    if args.loss == "weightedmse":
        cusloss = search_weights_loss(Ytrain.shape[0], Ytrain.shape[1], args.mag_factor * weight_samples[0])
    elif args.loss == "quad":
        cusloss = search_quadratic_loss(Ytrain.shape[0], Ytrain.shape[1], args.mag_factor * weight_samples[0].reshape(Ytrain.shape[1], args.quadrank), args.quadalpha)
    elif args.loss == "quad++":
        cusloss = search_direct_quadratic_loss(Ytrain.shape[0], Ytrain.shape[1], args.mag_factor * weight_samples[0].reshape(Ytrain.shape[1], Ytrain.shape[1], 4), args.quadalpha)

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

    if args.model == "xgb_search_decoupled":
        model = decoupledboosterwrapper(booster, Y_train[0].shape)
    else:
        model = treefromlodl(booster, Y_train[0].shape)

    if args.tree_check_logger:
        check_logger(cusloss.logger, args)


    metrics = print_metrics(model, problem, args.loss, get_loss_fn(args.evalloss, problem), "seed{}".format(args.seed), isTrain=False)

    return model, metrics










