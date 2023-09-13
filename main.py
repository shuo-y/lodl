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
from VMScheduling import VMScheduling
from VMScheduling_seq import VMSchedulingSeq
from RMAB import RMAB
from CubicTopK import CubicTopK
from models import model_dict
from losses import MSE, get_loss_fn
from utils import print_metrics, init_if_not_saved, move_to_gpu

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
        samples_filename_read=args.samples_read
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

def get_random_optDQ(Y, Y_aux, args):
    #   Document the value of a random guess
    objs_rand = []
    for _ in range(10):
        Z_rand = problem.get_decision(torch.rand_like(Y), aux_data=Y_aux, is_Train=False)
        objectives = problem.get_objective(Y, Z_rand, aux_data=Y_aux)
        objs_rand.append(objectives)
    randomdq = torch.stack(objs_rand).mean().item()
    print(f"Random Decision Quality: {randomdq}")

    #   Document the optimal value
    Z_fromtrue = problem.get_decision(Y, aux_data=Y_aux, is_Train=False)
    objectives = problem.get_objective(Y, Z_fromtrue, aux_data=Y_aux)
    obj = objectives.mean().item()

    if 'vmscheduling' in args.problem:
        print(f"VMScheduling opt if just using true prediction", obj)
        greedy_objs = problem.get_objective(Y, [None for _ in range(Y.shape[0])], Y_aux, dogreedy=True)
        # just work around here to check greedy algorithm we don't need the Z
        print("VMScheduling Greedy objs", greedy_objs.mean().item())
        return [randomdq, obj, greedy_objs.mean().item()]

    else:
        print(f"Decision Quality with true Y: {obj}")
        return [randomdq, obj]


def perf_metrics(args, problem, metrics):
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()
    print("X_train.shape {}".format(X_train.shape))
    print("Y_train.shape {}".format(Y_train.shape))
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    if args.measure_eval:
        Y_data = [Y_train, Y_test, Y_val]
        aux_data = [Y_train_aux, Y_test_aux, Y_val_aux]
        parts = ['train', 'test', 'val']
    else:
        Y_data = [Y_train, Y_test]
        aux_data = [Y_train_aux, Y_test_aux]
        parts = ['train', 'test']

    for (Y, aux, par_name) in zip(Y_data, aux_data, parts):
        print(par_name)
        metrics[par_name]['randomopt'] = get_random_optDQ(Y, aux, args)
        if args.problem == 'vmscheduling':
            metrics[par_name]['nordq'] = (metrics[par_name]['objective'] - metrics[par_name]['randomopt'][0])/ (metrics[par_name]['randomopt'][2] - metrics[par_name]['randomopt'][0])
        else:
            metrics[par_name]['nordq'] = (metrics[par_name]['objective'] - metrics[par_name]['randomopt'][0])/ (metrics[par_name]['randomopt'][1] - metrics[par_name]['randomopt'][0])


    sys.stdout.write("DQ_seed%d" % args.seed)
    for par_name in parts:
        sys.stdout.write(",%.12f" % metrics[par_name]['nordq'])
        sys.stdout.write(",%.12f" % metrics[par_name]['objective'])
        for ite in metrics[par_name]['randomopt']:
            sys.stdout.write(",%.12f" % ite)
    sys.stdout.write('\n')
    sys.stdout.flush()


def perf_multi_metrics(args, problem, metrics_list):
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()
    print("X_train.shape {}".format(X_train.shape))
    print("Y_train.shape {}".format(Y_train.shape))
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    if args.measure_eval:
        Y_data = [Y_train, Y_test, Y_val]
        aux_data = [Y_train_aux, Y_test_aux, Y_val_aux]
        parts = ['train', 'test', 'val']
    else:
        Y_data = [Y_train, Y_test]
        aux_data = [Y_train_aux, Y_test_aux]
        parts = ['train', 'test']

    random_opt_val - {}
    for (Y, aux, par_name) in zip(Y_data, aux_data, parts):
        print(par_name)
        random_opt_val[par_name]['randomopt'] = get_random_optDQ(Y, aux, args)

        # multiple metrics share the same random opt
        for metrics in metrics_list:
            metrics[par_name]['randomopt'] = random_opt_val[par_name]
            if args.problem == 'vmscheduling':
                metrics[par_name]['nordq'] = (metrics[par_name]['objective'] - metrics[par_name]['randomopt'][0])/ (metrics[par_name]['randomopt'][2] - metrics[par_name]['randomopt'][0])
            else:
                metrics[par_name]['nordq'] = (metrics[par_name]['objective'] - metrics[par_name]['randomopt'][0])/ (metrics[par_name]['randomopt'][1] - metrics[par_name]['randomopt'][0])


    for metrics in metrics_list:
        sys.stdout.write("DQ_seed%d" % args.seed)
        for par_name in parts:
            sys.stdout.write(",%.12f" % metrics[par_name]['nordq'])
            sys.stdout.write(",%.12f" % metrics[par_name]['objective'])
            for ite in metrics[par_name]['randomopt']:
                sys.stdout.write(",%.12f" % ite)
        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    # Get hyperparams from the command line
    # TODO: Separate main into folders per domain
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['budgetalloc', 'bipartitematching', 'cubic', 'rmab', 'portfolio', 'vmscheduling', 'vmschedulingseq'], default='portfolio')
    parser.add_argument('--loadnew', type=ast.literal_eval, default=False)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--iters', type=int, default=5000, help='used for original NN also search weights')
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--instances', type=int, default=400)
    parser.add_argument('--testinstances', type=int, default=200)
    parser.add_argument('--valfrac', type=float, default=0.5)
    parser.add_argument('--valfreq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['dense', 'xgb_decoupled', 'dense_multi', 'xgb_lodl', 'xgb_coupled', 'xgb_coupled_clf', 'xgb_search'], default='dense')
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
    parser.add_argument('--itempertrace', type=int, default=1000)
    #   Domain-specific: VMScheduling
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
    parser.add_argument('--search_estimators', type=int, default=10, help='use for cross entropy method search')
    parser.add_argument('--search_numsamples', type=int, default=100, help='use for cross entropy method search')
    parser.add_argument('--search_subsamples', type=int, default=10, help='use for cross entropy method search')
    parser.add_argument('--search_means', type=float, default=1, help='use for cross entropy method search the mean of the weights')
    parser.add_argument('--search_obj', type=str, default='cem_get_objective(problem, model, X_val, Y_val, Y_val_aux)', help='use for cross entropy method search the mean of the weights')
    parser.add_argument('--search_eval', type=str, default='[]')
    parser.add_argument('--num_estimators', type=int, default=10)
    parser.add_argument('--lodl_iter', type=int, default=10, help='if we want to train lodl multi rounds')
    parser.add_argument('--tree_method', type=str, default='hist', choices=['hist', 'gpu_hist', 'approx', 'auto', 'exact'])
    parser.add_argument('--tree_lambda', type=float, default=1, help='L2 normalization')
    parser.add_argument('--tree_eta', type=float, default=0.3, help='range is [0, 1]')
    parser.add_argument('--tree_alpha', type=float, default=0, help='L1 normalization')
    parser.add_argument('--tree_gamma', type=float, default=0, help='range is [0, +inf]')
    parser.add_argument('--tree_max_depth', type=int, default=6)
    parser.add_argument('--tree_min_child_weight', type=float, default=1, help='range is [0, +inf]')
    parser.add_argument('--tree_max_delta_step', type=float, default=0, help='range is [0, +inf]')
    parser.add_argument('--tree_subsample', type=float, default=1, help='range is [0, 1]')
    parser.add_argument('--tree_scale_pos_weight', type=float, default=1, help='ratio of positive and negative weights should be [0, +inf]')
    parser.add_argument('--tree_colsample_bytree', type=float, default=1, help='ratio of subsample (0, 1]')
    parser.add_argument('--tree_colsample_bylevel', type=float, default=1, help='ratio of subsample (0, 1]')
    parser.add_argument('--tree_colsample_bynode', type=float, default=1, help='ratio of subsample (0, 1]')
    parser.add_argument('--tree_check_logger', action='store_true', help='check the logger for tree')
    # Based on https://docs.python.org/3/library/argparse.html
    parser.add_argument('--lodlverbose', action='store_true')
    parser.add_argument('--weights_min', type=float, default=1e-3, help='minimum values of the weights')
    parser.add_argument('--mag_factor', type=float, default=1.0)
    parser.add_argument('--measure_eval', action='store_true')
    parser.add_argument('--samples_read', type=str, default='')
    parser.add_argument('--dumptree', action='store_true')
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--weights_vec', type=str, default='')
    parser.add_argument('--evalloss', type=str, choices=['mse', 'msesum', 'dense', 'weightedmse', 'weightedmse++', 'weightedce', 'weightedmsesum', 'dfl', 'quad', 'quad++', 'ce'], default='mse', help='use for evaluate the model with metrics')

    args = parser.parse_args()

    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")

    if args.problem == 'budgetalloc':
        problem = BudgetAllocation(num_train_instances = args.instances,
                                   num_test_instances = args.testinstances,
                                   num_targets = args.numtargets,
                                   num_items = args.numitems,
                                   budget = args.budget,
                                   num_fake_targets = args.fakefeatures,
                                   rand_seed = args.seed,
                                   val_frac = args.valfrac)
    elif args.problem == 'cubic':
        problem = CubicTopK(num_train_instances = args.instances,
                            num_test_instances = args.testinstances,
                            num_items = args.numitems,
                            budget = args.budget,
                            rand_seed = args.seed,
                            val_frac = args.valfrac)
    elif args.problem == 'bipartitematching':
        problem = BipartiteMatching(num_train_instances = args.instances,
                                    num_test_instances = args.testinstances,
                                    num_nodes = args.nodes,
                                    val_frac = args.valfrac,
                                    rand_seed = args.seed)
    elif args.problem == 'rmab':
        problem = RMAB(num_train_instances = args.instances,
                       num_test_instances = args.testinstances,
                       num_arms = args.numarms,
                       eval_method = args.eval,
                       min_lift = args.minlift,
                       budget = args.rmabbudget,
                       gamma = args.gamma,
                       num_features = args.numfeatures,
                       num_intermediate = args.scramblingsize,
                       num_layers = args.scramblinglayers,
                       noise_std = args.noisestd,
                       val_frac = args.valfrac,
                       rand_seed = args.seed)
    elif args.problem == 'portfolio':
        problem = PortfolioOpt(num_train_instances = args.instances,
                               num_test_instances = args.testinstances,
                               num_stocks = args.stocks,
                               alpha = args.stockalpha,
                               val_frac = args.valfrac,
                               rand_seed = args.seed)
    elif args.problem == 'vmscheduling':
        problem = VMScheduling(rand_seed=args.seed,
                               num_train=args.instances - int(args.valfrac * args.instances),
                               num_eval=int(args.valfrac * args.instances),
                               num_test=args.testinstances,
                               num_per_instance=args.itempertrace)
    elif args.problem == 'vmschedulingseq':
        problem = VMSchedulingSeq(rand_seed=args.seed,
                               num_train=args.instances - int(args.valfrac * args.instances),
                               num_eval=int(args.valfrac * args.instances),
                               num_test=args.testinstances,
                               num_per_instance=args.itempertrace)



    # Load an ML model to predict the parameters of the problem
    print(f"Building {args.model} Model...")
    if args.model == "xgb_decoupled":
        from train_xgb import train_xgb
        model, metrics = train_xgb(args, problem)
    elif args.model == "xgb_lodl" or args.model.startswith("xgb_coupled"):
        from train_xgb import train_xgb_lodl
        model, metrics = train_xgb_lodl(args, problem)
    elif args.model == "dense":
        model, metrics = train_dense(args, problem)
    elif args.model == "dense_multi":
        model, metrics = train_dense_multi(args, problem)
        # Document how well this trained model does
    elif args.model == "xgb_search":
        from train_xgb import train_xgb_search_weights
        model, metrics = train_xgb_search_weights(args, problem)

    perf_metrics(args, problem, metrics)

    print(args)


