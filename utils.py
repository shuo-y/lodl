import os
import pdb
from typing import Dict
import pandas as pd
import pickle
import torch
import inspect
import numpy as np
from itertools import repeat


def compute_stderror(vec: np.ndarray) -> float:
    popstd = vec.std()
    n = len(vec)
    return (popstd * np.sqrt(n / (n - 1.0))) / np.sqrt(n)

def sanity_check(vec: np.ndarray, msg: str) -> None:
    if (vec < 0).any():
        print(f"{msg}: check some negative value {vec}")


def perfrandomdq(problem, Y, Y_aux, trials):
    objs_rand = []
    for _ in range(trials):
        Z_rand = problem.get_decision(torch.rand_like(Y), aux_data=Y_aux, is_Train=False)
        objectives = problem.get_objective(Y, Z_rand, aux_data=Y_aux)
        objs_rand.append(objectives)

    randomdqs = torch.stack(objs_rand).mean(axis=0)
    return randomdqs


#def print_train_test(trainvaldl2st, testdl2st, trainvalsmac, testsmac, trainvaldlrand, testdlrand, trainvaldltrue, testdltrue, bltestdl):

def print_dq(dllist, namelist, cof):
    print("DQ,", end="")
    assert len(dllist) == len(namelist)
    for i in range(len(dllist)):
        dl = dllist[i]
        col = namelist[i]
        print(f"{col}, {col}_stderr, ", end="")
        print(f"{cof * dl.mean()}, {compute_stderror(cof * dl)}, ", end="")
    print()


def print_nor_dq(verbose, dllist, namelist, randdl, optdl):
    assert len(dllist) == len(namelist)
    for i in range(len(dllist)):
        dl = dllist[i]
        col = namelist[i]
        sanity_check(dl - optdl, f"{verbose}_{col}_nordq")

    print(verbose, end=", ")
    for i in range(len(dllist)):
        dl = dllist[i]
        col = namelist[i]
        nordq = (-dl + randdl)/ (-optdl + randdl)
        print(f"{col}nordq, {col}nordqstderr, ", end="")
        print(f"{nordq.mean()}, {compute_stderror(nordq)}, ", end="")
    print()

def print_nor_dq_filter0clip(verbose, dllist, namelist, randdl, optdl):
    assert len(dllist) == len(namelist)
    for i in range(len(dllist)):
        dl = dllist[i]
        col = namelist[i]
        sanity_check(dl - optdl, f"{verbose}_{col}_nordq")

    print(verbose, end=", ")
    res = []
    for i in range(len(dllist)):
        dl = dllist[i]
        col = namelist[i]
        nonzero = np.argwhere((randdl - optdl) != 0)
        nordq = (randdl[nonzero] - dl[nonzero])/ (randdl[nonzero] - optdl[nonzero])
        res.append(nordq)
        # Also clip between 0 1
        nordq = np.clip(nordq, 0, 1)
        print(f"{col}nordq, {col}nordqstderr, ", end="")
        print(f"{nordq.mean()}, {compute_stderror(nordq)}, ", end="")
    print()
    return res

def print_nor_dqagg(verbose, dllist, namelist, randdl, optdl):
    print(verbose, end=", ")
    for i in range(len(dllist)):
        dl = dllist[i]
        nordq = (randdl.mean() - dl.mean())/ (randdl.mean() - optdl.mean())
        print(f"agg_{namelist[i]}, {nordq},", end="")
    print("Not per instance normalized, aggregate normalized")


"""
    print("DQ, 2stagetrainvalobj, 2stagetestobj, "
          "2stagetrainvalobjstderr, 2stagetestobjstderr, "
          "smactrainvalobj, smactestobj, "
          "smactrainvalobjstderr, smactestobjstderr, "
          "randtrainvalobj, randtestobj, "
          "randtrainvalobjstderr, randtestobjstderr, "
          "truetrainvalobj, truetestobj, "
          "truetrainvalobjstderr, truetestobjstderr, ",
          "bltestdlobj, bltestdlobjstderr")
    print(f"DQ, {-1 * trainvaldl2st.mean()}, {-1 * testdl2st.mean()}, "
          f"{-1 * compute_stderror(trainvaldl2st)}, {-1 * compute_stderror(testdl2st)}, "
          f"{-1 * trainvalsmac.mean()}, {-1 * testsmac.mean()}, "
          f"{-1 * compute_stderror(trainvalsmac)}, {-1 * compute_stderror(testsmac)}, "
          f"{-1 * trainvaldlrand.mean()}, {-1 * testdlrand.mean()}, "
          f"{-1 * compute_stderror(trainvaldlrand)}, {-1 * compute_stderror(testdlrand)}, "
          f"{-1 * trainvaldltrue.mean()}, {-1 * testdltrue.mean()}, "
          f"{-1 * compute_stderror(trainvaldltrue)}, {-1 * compute_stderror(testdltrue)}, "
          f"{-1 * bltestdl.mean()}, {-1 * compute_stderror(bltestdl)}, ")

    nortest2stage = (-testdl2st + testdlrand)/(-testdltrue + testdlrand)
    nortestsmac = (-testsmac + testdlrand)/(-testdltrue + testdlrand)
    nortestbl = (-bltestdl + testdlrand)/(-testdltrue + testdlrand)

    print("NorDQ, 2stagetest, smactest, bltest, "
          "2stageteststderr, smacteststderr, blteststderr, ")
    print(f"NorDQ, {nortest2stage.mean()}, {nortestsmac.mean()}, {nortestbl.mean()}, "
          f"{compute_stderror(nortest2stage)}, {compute_stderror(nortestsmac)}, {compute_stderror(nortestbl)} ")
"""

def print_train_val_test(traindl2st, valdl2st, testdl2st, trainsmac, valsmac, testsmac, traindlrand, valdlrand, testdlrand, traindltrue, valdltrue, testdltrue, bltestdl):
    sanity_check(testdl2st - testdltrue, "test2st")
    sanity_check(testsmac - testdltrue, "testsmac")
    sanity_check(testdlrand - testdltrue, "testrand")
    sanity_check(bltestdl - testdltrue, "testbl")


    print("DQ, 2stagetrainobj, 2stagevalobj, 2stagetestobj, "
          "smactrainobj, smacvalobj, smactestobj, "
          "smactrainobjstderr, smacvalobjstderr, smactestobjstderr, "
          "randtrainobj, randvalobj, randtestobj, "
          "randtrainobjstderr, randvalobjstderr, randtestobjstderr, "
          "truetrainobj, truevalobj, truetestobj, "
          "truetrainobjstderr, truevalobjstderr, truetestobjstderr, "
          "bltestdlobj, bltestdlobjstderr")
    print(f"DQ, {-1 * traindl2st.mean()}, {-1 * valdl2st.mean()}, {-1 * testdl2st.mean()}, "
          f"{-1 * compute_stderror(traindl2st)}, {-1 * compute_stderror(valdl2st)}, {-1 * compute_stderror(testdl2st)}, "
          f"{-1 * trainsmac.mean()}, {-1 * valsmac.mean()}, {-1 * testsmac.mean()}, "
          f"{-1 * compute_stderror(trainsmac)}, {-1 * compute_stderror(valsmac)}, {-1 * compute_stderror(testsmac)}, "
          f"{-1 * traindlrand.mean()}, {-1 * valdlrand.mean()}, {-1 * testdlrand.mean()}, "
          f"{-1 * compute_stderror(traindlrand)}, {-1 * compute_stderror(valdlrand)}, {-1 * compute_stderror(testdlrand)}, "
          f"{-1 * traindltrue.mean()}, {-1 * valdltrue.mean()}, {-1 * testdltrue.mean()}, "
          f"{-1 * compute_stderror(traindltrue)}, {-1 * compute_stderror(valdltrue)}, {-1 * compute_stderror(testdltrue)}, "
          f"{-1 * bltestdl.mean()}, {-1 * compute_stderror(bltestdl)}, ")


    nortest2stage = (-testdl2st + testdlrand)/(-testdltrue + testdlrand)
    nortestsmac = (-testsmac + testdlrand)/(-testdltrue + testdlrand)
    nortestbl = (-bltestdl + testdlrand)/(-testdltrue + testdlrand)

    print("NorDQ, 2stagetest, smactest, bltest, "
          "2stageteststderr, smacteststderr, blteststderr, ")
    print(f"NorDQ, {nortest2stage.mean()}, {nortestsmac.mean()}, {nortestbl.mean()}, "
          f"{compute_stderror(nortest2stage)}, {compute_stderror(nortestsmac)}, {compute_stderror(nortestbl)} ")

def print_metrics(
    model,
    problem,
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
