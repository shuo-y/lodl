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
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    ## Squeeze to the last feature
    Xtrain = X_train.numpy().reshape(-1, X_train.shape[-1])
    Ytrain = Y_train.numpy().reshape(-1, Y_train.shape[-1])

    Xval = X_val.numpy().reshape(-1, X_val.shape[-1])
    Yval = Y_val.numpy().reshape(-1, Y_val.shape[-1])

    reg = xgb.XGBRegressor(tree_method='hist', n_estimators=args.num_estimators)


    reg.fit(Xtrain, Ytrain, eval_set=[(Xtrain, Ytrain), (Xval, Yval)])
    model = custom_tree(reg, np.prod(X_train[0].shape), np.prod(Y_train[0].shape), Y_train[0].shape)
    from utils import print_metrics
    from losses import get_loss_fn
    metrics = print_metrics(model, problem, args.loss, get_loss_fn('mse', problem), f"Iter {iter_idx},", isTrain=True)
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

class custom_loss():
    def __init__(self, ygold, grad_hess_fn):
        self.ygold = ygold
        self.grad_hess_fn = grad_hess_fn
        #print("initial custom_loss")
        #print(ygold[0])

    # Need use static method ?
    # https://www.digitalocean.com/community/tutorials/python-static-method

    def get_obj_fn(self):
        def obj(predt: np.ndarray, dtrain: xgb.DMatrix):
            predt = predt.reshape(self.ygold.shape)
            grad = np.zeros(self.ygold.shape).reshape(self.ygold.shape[0], np.prod(self.ygold.shape[1:]))
            hes = np.zeros(self.ygold.shape).reshape(self.ygold.shape[0], np.prod(self.ygold.shape[1:]))
            for i in range(self.ygold.shape[0]):
                 grad[i], hes[i] = self.grad_hess_fn(predt[i], self.ygold[i], 'train', i)
            grad = grad.flatten()
            hes = grad.flatten()
            return grad, hes
        return obj

def train_xgb_lodl(args, problem):
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    print(f"Loading {args.loss} Loss Function...")
    from losses import get_loss_fn

    loss_fn, grad_hess_fn = get_loss_fn(
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
        get_grad_hess=True,
    )
    import pdb
    #pdb.set_trace()

    # Based on some code from https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html

    cusloss = custom_loss(Y_train.detach().numpy(), grad_hess_fn)
    obj_fun = cusloss.get_obj_fn()

    Xtrain = X_train.numpy().reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    Ytrain = Y_train.numpy().reshape(Y_train.shape[0], np.prod(Y_train.shape[1:]))

    Xval = X_val.numpy().reshape(X_val.shape[0], np.prod(X_val.shape[1:]))
    Yval = Y_val.numpy().reshape(Y_val.shape[0], np.prod(Y_val.shape[1:]))

    #reg = xgb.XGBRegressor(tree_method='hist', n_estimators=args.num_estimators)
    Xy = xgb.DMatrix(Xtrain, Ytrain)

    booster = xgb.train(
            {"tree_method": "hist",
             "num_target": Ytrain.shape[1]
            },
        dtrain=Xy,
        num_boost_round = args.num_estimators,
        obj = obj_fun)

    model = treefromlodl(booster, Y_train[0].shape)
    from utils import print_metrics
    metrics = print_metrics(model, problem, args.loss, loss_fn, f"tree evaluate", isTrain=False)
    return model, metrics


"""
numpy code
https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
some debug log
(Pdb) (Y_train+ 0.01).shape
torch.Size([200, 50])
(Pdb) res = loss_fn((Y_train+ 0.01), Y_train, aux_data=Y_train_aux)
*** TypeError: surrogate_decision_quality() missing 2 required positional arguments: 'partition' and 'index'
(Pdb) res = loss_fn((Y_train+ 0.01), Y_train, aux_data=Y_train_aux, partition='train', index=0)

(Pdb) res
tensor([-0.0296, -0.0293, -0.0291, -0.0293, -0.0294, -0.0292, -0.0294, -0.0294,
        -0.0294, -0.0288, -0.0292, -0.0295, -0.0294, -0.0294, -0.0294, -0.0291,
        -0.0292, -0.0290, -0.0294, -0.0294, -0.0294, -0.0294, -0.0294, -0.0295,
        -0.0294, -0.0293, -0.0294, -0.0294, -0.0294, -0.0295, -0.0293, -0.0292,
        -0.0289, -0.0290, -0.0290, -0.0291, -0.0294, -0.0291, -0.0294, -0.0295,
        -0.0292, -0.0295, -0.0294, -0.0293, -0.0295, -0.0295, -0.0293, -0.0294,
        -0.0291, -0.0293, -0.0294, -0.0294, -0.0295, -0.0294, -0.0293, -0.0288,
        -0.0293, -0.0294, -0.0294, -0.0294, -0.0293, -0.0294, -0.0294, -0.0292,
        -0.0294, -0.0293, -0.0295, -0.0294, -0.0295, -0.0293, -0.0292, -0.0289,
        -0.0293, -0.0294, -0.0294, -0.0294, -0.0295, -0.0294, -0.0294, -0.0293,
        -0.0295, -0.0294, -0.0295, -0.0295, -0.0291, -0.0294, -0.0288, -0.0291,
        -0.0291, -0.0291, -0.0294, -0.0294, -0.0295, -0.0294, -0.0293, -0.0294,
        -0.0294, -0.0294, -0.0294, -0.0295, -0.0294, -0.0289, -0.0294, -0.0294,
        -0.0295, -0.0294, -0.0295, -0.0294, -0.0294, -0.0286, -0.0292, -0.0293,
        -0.0294, -0.0294, -0.0294, -0.0295, -0.0294, -0.0280, -0.0294, -0.0294,
        -0.0294, -0.0294, -0.0295, -0.0294, -0.0288, -0.0286, -0.0293, -0.0293,
        -0.0294, -0.0294, -0.0294, -0.0295, -0.0291, -0.0294, -0.0294, -0.0295,
        -0.0295, -0.0295, -0.0292, -0.0288, -0.0285, -0.0294, -0.0294, -0.0293,
        -0.0294, -0.0293, -0.0292, -0.0287, -0.0292, -0.0294, -0.0294, -0.0294,
        -0.0293, -0.0293, -0.0292, -0.0284, -0.0292, -0.0290, -0.0293, -0.0294,
        -0.0295, -0.0294, -0.0292, -0.0279, -0.0293, -0.0294, -0.0294, -0.0294,
        -0.0294, -0.0295, -0.0283, -0.0292, -0.0294, -0.0295, -0.0294, -0.0294,
        -0.0294, -0.0294, -0.0287, -0.0294, -0.0293, -0.0294, -0.0294, -0.0294,
        -0.0293, -0.0292, -0.0277, -0.0292, -0.0294, -0.0294, -0.0291, -0.0287,
        -0.0294, -0.0288, -0.0293, -0.0294, -0.0295, -0.0295, -0.0295, -0.0294],
       grad_fn=<SubBackward0>)
(Pdb) res.shape
torch.Size([200])
(Pdb) res = loss_fn((Y_train[0]+ 0.01), Y_train[0], aux_data=Y_train_aux[0], partition='train', index=0)
(Pdb) res.shape
torch.Size([1])
(Pdb) res
tensor([-0.0296], grad_fn=<SubBackward0>)
(Pdb) res = loss_fn((Y_train[1]+ 0.01), Y_train[1], aux_data=Y_train_aux[1], partition='train', index=1)
(Pdb) res.shape
torch.Size([1])
(Pdb) res = loss_fn((Y_train[1]+ 0.01), Y_train[1], aux_data=Y_train_aux[1], partition='train', index=200)
*** IndexError: list index out of range
(Pdb) res = loss_fn((Y_train[1]+ 0.01), Y_train[1], aux_data=Y_train_aux[1], partition='train', index=199)
(Pdb)


debug log
(Pdb) booster.predict(Xtrain)
*** TypeError: ('Expecting data to be a DMatrix object, got: ', <class 'numpy.ndarray'>)
(Pdb) booster.inplace_predict(Xtrain)
array([[0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
       ...,
       [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
       [0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5]], dtype=float32)
(Pdb) booster.inplace_predict(Xtrain).shape
(80, 50)
(Pdb) Ytrain.shape
(80, 50)
(Pdb) Y_train.shape
torch.Size([80, 5, 10])
(Pdb) booster.predict(Xtrain)



"""





