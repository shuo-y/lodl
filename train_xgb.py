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
    def __init__(self, ygold, loss_model_fn, loss_fn):
        self.ygold = ygold
        self.loss_model_fn = loss_model_fn
        self.loss_fn = loss_fn
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

    print("X_train.shape {}".format(X_train.shape))
    print("Y_train.shape {}".format(Y_train.shape))

    if args.model == "xgb_coupled":
        # 2stage xgboost coupled version
        print("Using native xgb.XGBRegressor")
        print("This option will ignore the args.loss")
        reg = xgb.XGBRegressor(tree_method='hist', n_estimators=args.num_estimators)
        reg.fit(Xtrain, Ytrain, eval_set=[(Xtrain, Ytrain)])
        dump_booster(reg.get_booster(), args)
        model = xgbwrapper(reg, Y_train[0].shape)
        from utils import print_metrics
        metrics = print_metrics(model, problem, args.loss, get_loss_fn("mse", problem), "seed{}".format(args.seed), isTrain=False)
        return model, metrics


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
    )
    import pdb
    #pdb.set_trace()

    # Based on some code from https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html

    cusloss = custom_loss(Y_train.detach().numpy(), loss_model_fn, loss_fn)
    obj_fun = cusloss.get_obj_fn()
    eval_fun = cusloss.get_eval_fn()



    #reg = xgb.XGBRegressor(tree_method='hist', n_estimators=args.num_estimators)
    Xy = xgb.DMatrix(Xtrain, Ytrain)
    Xyval = xgb.DMatrix(Xval, Yval)

    booster = xgb.train(
            {"tree_method": args.tree_method,
             "num_target": Ytrain.shape[1],
             "lambda": args.tree_lambda,
             "alpha": args.tree_alpha,
             "eta": args.tree_eta
            },
        dtrain=Xy,
        num_boost_round = args.num_estimators,
        obj = obj_fun,
        evals = [(Xy, "train")],
        custom_metric = eval_fun)
    dump_booster(booster, args)
    model = treefromlodl(booster, Y_train[0].shape)

    from utils import print_metrics
    metrics = print_metrics(model, problem, args.loss, get_loss_fn("mse", problem), "seed{}".format(args.seed), isTrain=False)
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










