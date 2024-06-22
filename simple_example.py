# coding: utf-8
#from pathlib import Path
# The example is from https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import lightgbm as lgb
#from losses import search_weights_loss, search_quadratic_loss, search_weights_directed_loss

print("Loading data...")
# load or create your dataset
#regression_example_dir = Path(__file__).absolute().parents[1] / "regression"
df_train = pd.read_csv("regression.train", header=None, sep="\t").to_numpy()
df_test = pd.read_csv("regression.test", header=None, sep="\t").to_numpy()

y_train = df_train[:, :2]
y_test = df_test[:, :2]
X_train = df_train[:, 2:]
X_test = df_test[:, 2:]

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

#gbm_def2 = lgb.LGBMRegressor(n_estimators=50, verbose=-1, random_state=0)
#gbm_def2.fit(X_train, y_train)
# Based on https://forecastegy.com/posts/lightgbm-multi-output-regression-classification-python/

lgb_model = lgb.LGBMRegressor(n_estimators=100)
lgb_multi = MultiOutputRegressor(lgb_model)
lgb_multi.fit(X_train, y_train)

import pdb
pdb.set_trace()

# specify your configurations as a dict
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "seed": 0
}
#    "metric": {"l2", "l1"},
#    "num_leaves": 31,
#    "learning_rate": 0.05,
#    "feature_fraction": 0.9,
#    "bagging_fraction": 0.8,
#    "bagging_freq": 5,
#    "verbose": 0,

print("Starting training...")
# train
gbm = lgb.train(
    params, lgb_train, num_boost_round=20, valid_sets=lgb_eval
)

print("Saving model...")
# save model to file
gbm.save_model("model.txt")

print("Starting predicting...")
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
#rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
#print(f"The RMSE of prediction is: {rmse_test}")

#
def grad_fn(predt, tdata):
    y = tdata.get_label()
    weights_pos = np.array([1.0 for _  in range(7000)])
    weights_neg = np.array([1.0 for _  in range(7000)])

    diff = (predt - y)
    posdiff = (diff >= 0) * diff
    negdiff = (diff < 0) * diff

    grad = (2 * (weights_pos * posdiff) + 2 * (weights_neg * negdiff))
    hess = (2 * (weights_pos * (diff >= 0)) + 2 * (weights_neg * (diff < 0)))

    grad = grad.flatten()
    grad = grad #/ len(grad)  # TODO check if normalized matters when the feature is unrelated?
    hess = hess.flatten()
    hess = hess #/ len(hess)
    return grad, hess
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor

#gbm_cus = lgb.LGBMRegressor(n_estimators=50, verbose=-1, objective=grad_fn, random_state=0)
#gbm_cus.fit(X_train, y_train)

# Check document
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor





gbm_cus3 = lgb.train({"boosting_type": "gbdt", "objective": grad_fn, "seed": 0}, lgb_train, num_boost_round=20, valid_sets=lgb_eval)

