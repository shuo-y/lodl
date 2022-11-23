import argparse
import torch
import random
import pdb
import matplotlib.pyplot as plt

from CubicTopK import CubicTopK
from models import model_dict
from losses import get_loss_fn

import xgboost as xgb

def train_xgb(problem):
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()

    reg = xgb.XGBRegressor(tree_method="hist", 
        n_estimators=10)
    X_train = X_train.squeeze().numpy()
    Y_train = Y_train.squeeze().numpy()

    X_val = X_val.squeeze().numpy()
    Y_val = Y_val.squeeze().numpy()

    X_test = X_test.squeeze().numpy()

    reg.fit(X_train, Y_train, eval_set=[(X_val, Y_val)])

    pred = reg.predict(X_test)
    pred = torch.tensor(pred)

    print("Benchmarking xgb ...")

    pred_train = torch.tensor(reg.predict(X_train))
    train_loss = np.mean([loss_fn(pred_it, Y_train_it).item() for pred_it, Y_train_it in zip(pred_train, Y_train)])
    print(f"train data mse score {train_loss.item()}")

    loss = np.mean([loss_fn(pred_it, Y_test_it).item() for pred_it, Y_test_it in zip(pred, Y_test)])
    print(f"\nAverage Test Loss on {args.loss} loss: {loss.item()}")

    #   In terms of problem objective
    Z_test = problem.get_decision(pred)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"\nTest Decision Quality: {objectives.mean().item()}")

    #   Document the value of a random guess
    Z_test = problem.get_decision(torch.rand_like(Y_test))
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Random Decision Quality: {objectives.mean().item()}")

    Z_test = problem.get_decision(torch.rand_like(Y_test))
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Random Decision Quality: {objectives.mean().item()}")


    #   Document the optimal value
    Z_test = problem.get_decision(Y_test)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Optimal Decision Quality: {objectives.mean().item()}")
    print()


def train_xgb_flatten(problem):
    print(f"Train xgb with flatten data")

    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()

    reg = xgb.XGBRegressor(tree_method="hist",
        n_estimators=10)
    X_train = X_train.flatten().unsqueeze(-1).numpy()
    Y_train = Y_train.flatten().unsqueeze(-1).numpy()

    X_val = X_val.flatten().unsqueeze(-1).numpy()
    Y_val = Y_val.flatten().unsqueeze(-1).numpy()

    X_test = X_test.flatten().unsqueeze(-1).numpy()
    Y_test = Y_test.flatten().unsqueeze(-1)

    reg.fit(X_train, Y_train, eval_set=[(X_val, Y_val)])

    pred = reg.predict(X_test)
    pred = torch.tensor(pred)

    print("Benchmarking xgb ...")

    pred_train = torch.tensor(reg.predict(X_train))
    train_loss = np.mean([loss_fn(pred_it, Y_train_it).item() for pred_it, Y_train_it in zip(pred_train, Y_train)])
    print(f"train data mse score {train_loss.item()}")

    loss = np.mean([loss_fn(pred_it, Y_test_it).item() for pred_it, Y_test_it in zip(pred, Y_test)])
    print(f"\nAverage Test Loss on {args.loss} loss: {loss.item()}")

    "change the data back to original dimension, looks like cheating..."
    Y_test = Y_test.view(problem.num_test_instances, problem.num_items)
    pred = pred.view(problem.num_test_instances, problem.num_items)

    #   In terms of problem objective
    Z_test = problem.get_decision(pred)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"\nTest Decision Quality: {objectives.mean().item()}")

    #   Document the value of a random guess
    Z_test = problem.get_decision(torch.rand_like(Y_test))
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Random Decision Quality: {objectives.mean().item()}")

    Z_test = problem.get_decision(torch.rand_like(Y_test))
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Random Decision Quality: {objectives.mean().item()}")


    #   Document the optimal value
    Z_test = problem.get_decision(Y_test)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Optimal Decision Quality: {objectives.mean().item()}")
    print()


# Based on https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html#sphx-glr-python-examples-multioutput-regression-py
def custom_rmse_model(problem) -> None:
    """Train using Python implementation of Squared Error."""

    # As the experimental support status, custom objective doesn't support matrix as
    # gradient and hessian, which will be changed in future release.
    from typing import Dict, Tuple, List

    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        """Compute the gradient squared error."""
        y = dtrain.get_label().reshape(predt.shape)
        return (predt - y).reshape(y.size)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        """Compute the hessian for squared error."""
        return np.ones(predt.shape).reshape(predt.size)

    def squared_log(
        predt: np.ndarray, dtrain: xgb.DMatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad = gradient(predt, dtrain)
        hess = hessian(predt, dtrain)
        return grad, hess

    def rmse(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        y = dtrain.get_label().reshape(predt.shape)
        v = np.sqrt(np.sum(np.power(y - predt, 2)))
        return "PyRMSE", v

    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()

    X_train = X_train.flatten().unsqueeze(-1).numpy()
    Y_train = Y_train.flatten().unsqueeze(-1).numpy()

    Xy = xgb.DMatrix(X_train, Y_train)
    results: Dict[str, Dict[str, List[float]]] = {}
    # Make sure the `num_target` is passed to XGBoost when custom objective is used.
    # When builtin objective is used, XGBoost can figure out the number of targets
    # automatically.
    import pdb
    pdb.set_trace()

    booster = xgb.train(
        {
            "tree_method": "hist",
            "num_target": Y_test.shape[-1],
        },
        dtrain=Xy,
        num_boost_round=100,
        obj=squared_log,
        evals=[(Xy, "Train")],
        evals_result=results,
        custom_metric=rmse,
    )

    X_test = X_test.squeeze().numpy()
    pred = booster.inplace_predict(X_test)

    pred = torch.tensor(pred)

    print("Benchmarking xgb ...")

    pred_train = torch.tensor(reg.predict(X_train))
    train_loss = np.mean([loss_fn(pred_it, Y_train_it).item() for pred_it, Y_train_it in zip(pred_train, Y_train)])
    print(f"train data mse score {train_loss.item()}")

    loss = np.mean([loss_fn(pred_it, Y_test_it).item() for pred_it, Y_test_it in zip(pred, Y_test)])
    print(f"\nAverage Test Loss on {args.loss} loss: {loss.item()}")

    #   In terms of problem objective
    Z_test = problem.get_decision(pred)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"\nTest Decision Quality: {objectives.mean().item()}")

    #   Document the value of a random guess
    Z_test = problem.get_decision(torch.rand_like(Y_test))
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Random Decision Quality: {objectives.mean().item()}")

    Z_test = problem.get_decision(torch.rand_like(Y_test))
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Random Decision Quality: {objectives.mean().item()}")


    #   Document the optimal value
    Z_test = problem.get_decision(Y_test)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Optimal Decision Quality: {objectives.mean().item()}")
    print()

if __name__ == '__main__':
    # Get hyperparams from the command line
    # TODO: Do this for all the domains together
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--num_train_instances', type=int, default=100)
    parser.add_argument('--num_test_instances', type=int, default=100)
    parser.add_argument('--instances', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['dense'], default='dense')
    parser.add_argument('--loss', type=str, choices=['mse', 'msesum', 'dense', 'weightedmse', 'weightedmsesum', 'dfl', 'quad'], default='mse')
    parser.add_argument('--fakefeatures', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--sampling', type=str, choices=['random', 'numerical_jacobian', 'random_jacobian'], default='random_jacobian')
    #   Domain-specific
    parser.add_argument('--numitems', type=int, default=100)
    parser.add_argument('--budget', type=int, default=2)
    args = parser.parse_args()
    print(args)

    # Load problem
    print("Loading Problem...")
    problem = CubicTopK(
        num_train_instances=args.num_train_instances,
        num_test_instances=args.num_test_instances,
        budget=args.budget,
        rand_seed=0
    )

    problem_xgb = CubicTopK(
        num_train_instances=args.num_train_instances,
        num_test_instances=args.num_test_instances,
        budget=args.budget,
        rand_seed=0
    )

    problem_xgb_2 = CubicTopK(
        num_train_instances=args.num_train_instances,
        num_test_instances=args.num_test_instances,
        budget=args.budget,
        rand_seed=0
    )


    # Load a loss function to train the ML model on
    #   TODO: Abstract over this loss for the proposed method
    #   TODO: Figure out loss function "type" for mypy type checking
    print("Loading Loss Function...")
    loss_fn = get_loss_fn(args.loss, problem)

    # Load an ML model to predict the parameters of the problem
    #   TODO: Abstract over models? What should model builder look like in general?
    print("Building Model...")
    model_builder = model_dict[args.model]
    model = model_builder(
        num_features=1,
        num_targets=1,
        num_layers=args.layers,
        intermediate_size=500,
        output_activation=None,
    )
    #   TODO: Add ways to modify lr, etc?
    optimizer = torch.optim.Adam(model.parameters())

    # Train neural network with a given loss function
    #   TODO: Add early stopping
    print(f"Training {args.model} model on {args.loss} loss...")
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    for iter_idx in range(args.iters):
        losses = []
        for i in random.sample(range(len(X_train)), min(args.batchsize, len(X_train))):
            pred = model(X_train[i])
            losses.append(loss_fn(pred, Y_train[i], index=i))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print metrics
    #   TODO: Save the learned model

    # Document how well this trained model does
    print("Benchmarking Model...")
    X_test, Y_test, Y_test_aux = problem.get_test_data()
    pred = model(X_test).squeeze()

    #   Plot predictions on test data
    plt.hist(pred.flatten().tolist(), bins=100, alpha=0.5, label='pred')
    plt.hist(Y_test.flatten().tolist(), bins=100, alpha=0.5, label='true')
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig("cubictopk.svg")

    #   In terms of loss function
    if args.loss in ["mse", "msesum"]:
        import numpy as np
        pred_train = model(X_train).squeeze()
        train_loss = np.mean([loss_fn(pred_it, Y_train_it).item() for pred_it, Y_train_it in zip(pred_train, Y_train)])
        print(f"Average Train loss on {args.loss} loss: {loss.item()}")
        loss = np.mean([loss_fn(pred_it, Y_test_it).item() for pred_it, Y_test_it in zip(pred, Y_test)])
        print(f"\nAverage Test Loss on {args.loss} loss: {loss.item()}")

    #   In terms of problem objective
    Z_test = problem.get_decision(pred)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"\nTest Decision Quality: {objectives.mean().item()}")

    #   Document the value of a random guess
    Z_test = problem.get_decision(torch.rand_like(Y_test))
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Random Decision Quality: {objectives.mean().item()}")

    #   Document the optimal value
    Z_test = problem.get_decision(Y_test)
    objectives = problem.get_objective(Y_test, Z_test)
    print(f"Optimal Decision Quality: {objectives.mean().item()}")
    print()

    print("XGB")
    train_xgb(problem_xgb)
    #custom_rmse_model(problem_xgb_2)


