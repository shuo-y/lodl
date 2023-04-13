from functools import partial
import os
import sys

# Makes sure hashes are consistent
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import matplotlib.pyplot as plt
import ast
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import random
import pdb
from copy import deepcopy

from BudgetAllocation import BudgetAllocation
from BipartiteMatching import BipartiteMatching
from PortfolioOpt import PortfolioOpt
from RMAB import RMAB
from CubicTopK import CubicTopK
from models import model_dict
from losses import MSE, get_loss_fn
from utils import print_metrics, init_if_not_saved, move_to_gpu

def train_dense_multi(args, problem):
    # Load a loss function to train the ML model on
    #   TODO: Figure out loss function "type" for mypy type checking. Define class/interface?
    # Get data
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()

    ipdim, opdim = problem.get_modelio_shape()
    model_builder = model_dict[args.model]
    model = model_builder(
        num_features=ipdim,
        num_targets=opdim,
        num_layers=args.layers,
        intermediate_size=500,
        output_activation=problem.get_output_activation(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Loading {args.loss} Loss Function...")
    for i in range(args.lodl_iter):
        model.eval()
        loss_fn = get_loss_fn(
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
            train_model=model,
        )
        model.train()

        # Train neural network with a given loss function
        print(f"Training {args.model} model on {args.loss} loss...")
        #   Move everything to GPU, if available
        #if torch.cuda.is_available():
        #    move_to_gpu(problem)
        #    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #    model = model.to(device)


        best = (float("inf"), None)
        time_since_best = 0
        for iter_idx in range(args.iters):
            # Check metrics on val set
            if iter_idx % args.valfreq == 0:
                # Compute metrics
                metrics = print_metrics(model, problem, args.loss, loss_fn, f"Iter {iter_idx},", isTrain=True)

                # Save model if it's the best one
                if best[1] is None or metrics['val']['loss'] < best[0]:
                    best = (metrics['val']['loss'], deepcopy(model))
                    time_since_best = 0

                # Stop if model hasn't improved for patience steps
                if args.earlystopping and time_since_best > args.patience:
                    break

            # Learn
            losses = []
            for i in random.sample(range(len(X_train)), min(args.batchsize, len(X_train))):
                pred = model(X_train[i]).squeeze()
                losses.append(loss_fn(pred, Y_train[i], aux_data=Y_train_aux[i], partition='train', index=i))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time_since_best += 1

        if args.earlystopping:
            model = best[1]

    metrics = print_metrics(model, problem, args.loss, loss_fn, "final", skiptestloss=True)
    return model, metrics



def train_dense(args, problem):
    # Load a loss function to train the ML model on
    #   TODO: Figure out loss function "type" for mypy type checking. Define class/interface?
    print(f"Loading {args.loss} Loss Function...")
    loss_fn = get_loss_fn(
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
    )
    # Get data
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()

    ipdim, opdim = problem.get_modelio_shape()
    model_builder = model_dict[args.model]
    model = model_builder(
        num_features=ipdim,
        num_targets=opdim,
        num_layers=args.layers,
        intermediate_size=500,
        output_activation=problem.get_output_activation(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train neural network with a given loss function
    print(f"Training {args.model} model on {args.loss} loss...")
    #   Move everything to GPU, if available
    #if torch.cuda.is_available():
    #    move_to_gpu(problem)
    #    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #    model = model.to(device)


    best = (float("inf"), None)
    time_since_best = 0
    for iter_idx in range(args.iters):
        # Check metrics on val set
        if iter_idx % args.valfreq == 0:
            # Compute metrics
            metrics = print_metrics(model, problem, args.loss, loss_fn, f"Iter {iter_idx},", isTrain=True)

            # Save model if it's the best one
            if best[1] is None or metrics['val']['loss'] < best[0]:
                best = (metrics['val']['loss'], deepcopy(model))
                time_since_best = 0

            # Stop if model hasn't improved for patience steps
            if args.earlystopping and time_since_best > args.patience:
                break

        # Learn
        losses = []
        for i in random.sample(range(len(X_train)), min(args.batchsize, len(X_train))):
            pred = model(X_train[i]).squeeze()
            losses.append(loss_fn(pred, Y_train[i], aux_data=Y_train_aux[i], partition='train', index=i))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_since_best += 1

    if args.earlystopping:
        model = best[1]

    print("\nBenchmarking Model...")
    # Print final metrics
    metrics = print_metrics(model, problem, args.loss, loss_fn, "Final_seed{}".format(args.seed))

    return model, metrics

def get_random_optDQ(Y, Y_aux):
    #   Document the value of a random guess
    objs_rand = []
    for _ in range(10):
        Z_rand = problem.get_decision(torch.rand_like(Y), aux_data=Y_aux, isTrain=False)
        objectives = problem.get_objective(Y, Z_rand, aux_data=Y_aux)
        objs_rand.append(objectives)
    randomdq = torch.stack(objs_rand).mean().item()
    print(f"Random Decision Quality: {randomdq}")

    #   Document the optimal value
    Z_opt = problem.get_decision(Y, aux_data=Y_aux, isTrain=False)
    objectives = problem.get_objective(Y, Z_opt, aux_data=Y_aux)
    optimaldq = objectives.mean().item()
    print(f"Optimal Decision Quality: {optimaldq}")
    return randomdq, optimaldq


if __name__ == '__main__':
    # Get hyperparams from the command line
    # TODO: Separate main into folders per domain
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['budgetalloc', 'bipartitematching', 'cubic', 'rmab', 'portfolio'], default='portfolio')
    parser.add_argument('--loadnew', type=ast.literal_eval, default=False)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--instances', type=int, default=400)
    parser.add_argument('--testinstances', type=int, default=200)
    parser.add_argument('--valfrac', type=float, default=0.5)
    parser.add_argument('--valfreq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['dense', 'xgb_decoupled', 'dense_multi', 'xgb_lodl', 'xgb_coupled'], default='dense')
    parser.add_argument('--loss', type=str, choices=['mse', 'msesum', 'dense', 'weightedmse', 'weightedmse++', 'weightedce', 'weightedmsesum', 'dfl', 'quad', 'quad++', 'ce'], default='mse')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=1000)
    #   Domain-specific: BudgetAllocation or CubicTopK
    parser.add_argument('--budget', type=int, default=1)
    parser.add_argument('--numitems', type=int, default=50)
    #   Domain-specific: BudgetAllocation
    parser.add_argument('--numtargets', type=int, default=10)
    parser.add_argument('--fakefeatures', type=int, default=0)
    #   Domain-specific: RMAB
    parser.add_argument('--rmabbudget', type=int, default=1)
    parser.add_argument('--numarms', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--minlift', type=float, default=0.2)
    parser.add_argument('--scramblinglayers', type=int, default=3)
    parser.add_argument('--scramblingsize', type=int, default=64)
    parser.add_argument('--numfeatures', type=int, default=16)
    parser.add_argument('--noisestd', type=float, default=0.5)
    parser.add_argument('--eval', type=str, choices=['exact', 'sim'], default='exact')
    #   Domain-specific: BipartiteMatching
    parser.add_argument('--nodes', type=int, default=10)
    #   Domain-specific: PortfolioOptimization
    parser.add_argument('--stocks', type=int, default=50)
    parser.add_argument('--stockalpha', type=float, default=0.1)
    #   Decision-Focused Learning
    parser.add_argument('--dflalpha', type=float, default=1.)
    #   Learned-Loss
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--sampling', type=str, choices=['random', 'random_flip', 'random_uniform', 'numerical_jacobian', 'random_jacobian', 'random_hessian', 'sample_iter'], default='random')
    parser.add_argument('--samplingstd', type=float)
    parser.add_argument('--numsamples', type=int, default=5000)
    parser.add_argument('--losslr', type=float, default=0.01)
    #       Approach-Specific: Quadratic
    parser.add_argument('--quadrank', type=int, default=20)
    parser.add_argument('--quadalpha', type=float, default=0)
    parser.add_argument('--num_estimators', type=int, default=10)
    parser.add_argument('--lodl_iter', type=int, default=10, help='if we want to train lodl multi rounds')
    parser.add_argument('--tree_method', type=str, default='hist', choices=['hist', 'approx', 'auto', 'exact'])
    parser.add_argument('--tree_lambda', type=float, default=1)
    parser.add_argument('--tree_eta', type=float, default=0.3)
    parser.add_argument('--tree_alpha', type=float, default=0)
    # Based on https://docs.python.org/3/library/argparse.html
    parser.add_argument('--lodlverbose', action='store_true')
    args = parser.parse_args()

    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")
    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
    if args.problem == 'budgetalloc':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_targets': args.numtargets,
                            'num_items': args.numitems,
                            'budget': args.budget,
                            'num_fake_targets': args.fakefeatures,
                            'rand_seed': args.seed,
                            'val_frac': args.valfrac,}
        problem = init_problem(BudgetAllocation, problem_kwargs)
    elif args.problem == 'cubic':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_items': args.numitems,
                            'budget': args.budget,
                            'rand_seed': args.seed,
                            'val_frac': args.valfrac,}
        problem = init_problem(CubicTopK, problem_kwargs)
    elif args.problem == 'bipartitematching':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_nodes': args.nodes,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
        problem = init_problem(BipartiteMatching, problem_kwargs)
    elif args.problem == 'rmab':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_arms': args.numarms,
                            'eval_method': args.eval,
                            'min_lift': args.minlift,
                            'budget': args.rmabbudget,
                            'gamma': args.gamma,
                            'num_features': args.numfeatures,
                            'num_intermediate': args.scramblingsize,
                            'num_layers': args.scramblinglayers,
                            'noise_std': args.noisestd,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
        problem = init_problem(RMAB, problem_kwargs)
    elif args.problem == 'portfolio':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'num_stocks': args.stocks,
                            'alpha': args.stockalpha,
                            'val_frac': args.valfrac,
                            'rand_seed': args.seed,}
        problem = init_problem(PortfolioOpt, problem_kwargs)



    # Load an ML model to predict the parameters of the problem
    print(f"Building {args.model} Model...")
    if args.model == "xgb_decoupled":
        from train_xgb import train_xgb
        model, metrics = train_xgb(args, problem)
    elif args.model == "xgb_lodl" or args.model == "xgb_coupled":
        from train_xgb import train_xgb_lodl
        model, metrics = train_xgb_lodl(args, problem)
    elif args.model == "dense":
        model, metrics = train_dense(args, problem)
    elif args.model == "dense_multi":
        model, metrics = train_dense_multi(args, problem)
        # Document how well this trained model does


    X_train, Y_train, Y_train_aux = problem.get_train_data()
    print("X_train.shape {}".format(X_train.shape))
    print("Y_train.shape {}".format(Y_train.shape))

    print("train set")
    trainrandomdq, trainoptimaldq = get_random_optDQ(Y_train, Y_train_aux)
    trainnordq = (metrics['train']['objective'] - trainrandomdq)/(trainoptimaldq - trainrandomdq)
    print("(Only this run) Normalize DQ on train set: %.12f" % trainnordq)

    _, Y_val, Y_val_aux = problem.get_val_data()
    print("eval set")
    valrandomdq, valoptimaldq = get_random_optDQ(Y_val, Y_val_aux)
    valnordq = (metrics['val']['objective'] - valrandomdq)/(valoptimaldq - valrandomdq)
    print("(Only this run) Normalize DQ on eval set: %.12f" % valnordq)

    print("test set")
    X_test, Y_test, Y_test_aux = problem.get_test_data()
    print("X_test.shape {}".format(X_test.shape))
    print("Y_test.shape {}".format(Y_test.shape))
    testrandomdq, testoptimaldq = get_random_optDQ(Y_test, Y_test_aux)
    testnordq = (metrics['test']['objective'] - testrandomdq)/(testoptimaldq - testrandomdq)
    print("(Only this run) Normalize DQ on test set: %.12f" % testnordq)

    print("DQ_seed%d,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f" % (args.seed, trainnordq, trainrandomdq, trainoptimaldq, valnordq, valrandomdq, valoptimaldq, testnordq, testrandomdq, testoptimaldq))
    print(args)

    # pdb.set_trace()

    # #   Plot predictions on test data
    # plt.scatter(Y_test.sum(dim=-1).flatten().detach().tolist(), pred.sum(dim=-1).flatten().detach().tolist(), )
    # plt.title(args.loss)
    # plt.xlabel("True")
    # plt.ylabel("Predicted")
    # plt.xlim([0, 0.5])
    # plt.ylim([0, 0.5])
    # plt.show()
