import xgboost as xgb
import torch

class custom_tree:
    # Make instance callable based on https://medium.com/swlh/callables-in-python-how-to-make-custom-instance-objects-callable-too-516d6eaf0c8d
    def __init__(self, reg):
        self.reg = reg

    def __call__(self, X: torch.Tensor):
        Xi = X.numpy().reshape(-1, X.shape[-1])
        Yi = self.reg.predict(Xi)
        Y = torch.tensor(Yi)
        Y = torch.reshape(Y, X.shape)
        return Y

def train_xgb(args, problem):
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()


    Xtrain = X_train.numpy().reshape(-1, X_train.shape[-1])
    Ytrain = Y_train.numpy().reshape(-1, Y_train.shape[-1])

    reg = xgb.XGBRegressor(tree_method='hist', n_estimators=10)
    ### Warning? Parameters: { "tree_mothod" } might not be used.

    #This could be a false alarm, with some parameters getting used by language bindings but
    #then being mistakenly passed down to XGBoost core, or some parameter actually being used
    #but getting flagged wrongly here. Please open an issue if you find any such cases.

    reg.fit(Xtrain, Ytrain, eval_set=[(Xtrain, Ytrain)])

    model = custom_tree(reg)
    return model
