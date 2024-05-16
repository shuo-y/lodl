import os
import pdb
from typing import Dict
import pandas as pd
import pickle
import torch
import inspect
from itertools import repeat


def perfrandomdq(problem, Y, Y_aux, trials):
    objs_rand = []
    for _ in range(trials):
        Z_rand = problem.get_decision(torch.rand_like(Y), aux_data=Y_aux, is_Train=False)
        objectives = problem.get_objective(Y, Z_rand, aux_data=Y_aux)
        objs_rand.append(objectives)

    import pdb
    pdb.set_trace()
    randomdqs = torch.stack(objs_rand).mean(axis=0)
    return randomdqs

def print_metrics(
    model,
    problem,
    loss_type,
    loss_fn,
    prefix="",
    isTrain=False,
):
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    if isTrain == False:
        X_test, Y_test, Y_test_aux = problem.get_test_data()
        datasets = [(X_train, Y_train, Y_train_aux, "train"), (X_val, Y_val, Y_val_aux, "val"), (X_test, Y_test, Y_test_aux, "test")]
    else:
        datasets = [(X_train, Y_train, Y_train_aux, "train"), (X_val, Y_val, Y_val_aux, "val")]
    # print(f"Current model parameters: {[param for param in model.parameters()]}")
    metrics = {}
    print("metrics use loss function: %s" % loss_fn.__name__)

    for Xs, Ys, Ys_aux, partition in datasets:
        # Choose whether we should use train or test 
        isTrain = (partition=="train") and (prefix != "Final")

        # Decision Quality
        pred = model(Xs).squeeze()
        Zs_pred = problem.get_decision(pred, aux_data=Ys_aux, isTrain=isTrain)
        objectives = problem.get_objective(Ys, Zs_pred, aux_data=Ys_aux)

        objective = objectives.mean().item()
        # Loss and Error
        losses = []
        for i in range(len(Xs)):
            # Surrogate Loss
            pred = model(Xs[i][None]).squeeze()
            losses.append(loss_fn(pred, Ys[i], aux_data=Ys_aux[i], partition=partition, index=i))
            if None in losses:
                print("This loss function does not support {} partition".format(partition))
                break

        if None in losses:
            metrics[partition] = {"objective": objective}
            continue
        losses = torch.stack(losses).flatten()
        loss = losses.mean().item()
        mae = torch.nn.L1Loss()(losses, -objectives).item()

        metrics[partition] = {"objective": objective, "loss": loss, "mae": mae, "objs": objectives}

    import sys
    sys.stdout.write(prefix)
    for par in metrics:
        sys.stdout.write(",%.12f" % metrics[par]["objective"])
        sys.stdout.write(",%.12f" % metrics[par]["loss"])
        sys.stdout.write(",%.12f" % metrics[par]["mae"])
    sys.stdout.write("\n")
    return metrics

def starmap_with_kwargs(pool, fn, args_iter, kwargs):
    args_for_starmap = zip(repeat(fn), args_iter, repeat(kwargs))
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

def gather_incomplete_left(tensor, I):
    return tensor.gather(I.ndim, I[(...,) + (None,) * (tensor.ndim - I.ndim)].expand((-1,) * (I.ndim + 1) + tensor.shape[I.ndim + 1:])).squeeze(I.ndim)

def trim_left(tensor):
    while tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor

class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.shape[:-1]
        shape = (*batch_size, *self.shape)
        out = input.view(shape)
        return out

def solve_lineqn(A, b, eps=1e-5):
    try:
        result = torch.linalg.solve(A, b)
    except RuntimeError:
        print(f"WARNING: The matrix was singular")
        result = torch.linalg.solve(A + eps * torch.eye(A.shape[-1]), b)
    return result

def move_to_gpu(problem):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for key, value in inspect.getmembers(problem, lambda a:not(inspect.isroutine(a))):
        if isinstance(value, torch.Tensor):
            problem.__dict__[key] = value.to(device)
