import sys
import argparse
import ast
import torch
import random


def get_random_optDQ(mpart, Y, Y_aux, args):
    #   Document the value of a random guess
    objs_rand = []
    for _ in range(10):
        Z_rand = problem.get_decision(torch.rand_like(Y), aux_data=Y_aux, is_Train=False)
        objectives = problem.get_objective(Y, Z_rand, aux_data=Y_aux)
        objs_rand.append(objectives)
    randomdqs = torch.stack(objs_rand).mean(axis=0)
    print(f"Random Decision Quality: {randomdqs.mean().item()}")

    #   Document the optimal value
    Z_fromtrue = problem.get_decision(Y, aux_data=Y_aux, is_Train=False)
    true_objs = problem.get_objective(Y, Z_fromtrue, aux_data=Y_aux)

    nordq = torch.zeros(len(randomdqs))
    assert len(mpart["objs"]) == len(randomdqs)
    assert len(mpart["objs"]) == len(true_objs)

    if 'vmscheduling' in args.problem:
        print(f"VMScheduling opt if just using true prediction", true_objs.mean().item())
        greedy_objs = problem.get_objective(Y, [None for _ in range(Y.shape[0])], Y_aux, dogreedy=True)
        assert len(mpart["objs"]) == len(greedy_objs)
        # just work around here to check greedy algorithm we don't need the Z
        print("VMScheduling Greedy objs", greedy_objs.mean().item())
        for i in range(len(randomdqs)):
            nordq[i] = (mpart["objs"][i] - randomdqs[i])/(greedy_objs[i] - randomdqs[i])
        mpart['randomopt'] = [nordq.mean().item(), randomdqs.mean().item(), true_objs.mean().item(), greedy_objs.mean().item()]

    else:
        for i in range(len(randomdqs)):
            nordq[i] = (mpart["objs"][i] - randomdqs[i])/(true_objs[i] - randomdqs[i])
        print(f"Decision Quality with true Y: {true_objs.mean().item()}")
        if any(nordq.isnan()):
            print(f"Some nordq is nan dropped...")
            nordq = nordq[~nordq.isnan()]
        mpart['randomopt'] = [nordq.mean().item(), randomdqs.mean().item(), true_objs.mean().item()]
        # return norDQ  realDQ randomDQ optDQ


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
        #metrics[par_name]['randomopt'] = get_random_optDQ(Y, aux, args)
        get_random_optDQ(metrics[par_name], Y, aux, args)

    sys.stdout.write("DQ_seed%d" % args.seed)
    for par_name in parts:
        for ite in metrics[par_name]['randomopt']:
            sys.stdout.write(",%.12f" % ite)
    sys.stdout.write(",obj_loss_seed%d" % args.seed)
    for par_name in parts:
        sys.stdout.write(",%.12f" % metrics[par_name]['objective'])
        sys.stdout.write(",%.12f" % metrics[par_name]['loss'])
        sys.stdout.write(",%.12f" % metrics[par_name]['mae'])

    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    # Get hyperparams from the command line
    # TODO: Separate main into folders per domain
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['budgetalloc', 'bipartitematching', 'cubic', 'rmab', 'portfolio', 'vmscheduling', 'vmschedulingseq', 'shortestpath'], default='portfolio')
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
    parser.add_argument('--model', type=str, choices=['dense', 'dense_coupled', 'xgb_decoupled', 'xgb_lodl', 'xgb_lodl_decoupled', 'xgb_coupled', 'xgb_coupled_clf', 'xgb_search', 'xgb_search_decoupled', 'xgb_ngopt'], default='dense')
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
    parser.add_argument('--search_means', type=float, default=1, help='use for cross entropy method search the initial mean')
    parser.add_argument('--search_covs', type=float, default=1, help='use for cross entropy method search the initial covs')
    parser.add_argument('--restart_rounds', type=int, default=1, help='the restart round for CEM search')
    parser.add_argument('--num_estimators', type=int, default=10)
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
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--weights_min', type=float, default=1e-3, help='minimum values of the weights')
    parser.add_argument('--mag_factor', type=float, default=1.0, help='this is for magnifier the weights')
    parser.add_argument('--measure_eval', action='store_true')
    parser.add_argument('--samples_read', type=str, default='')
    parser.add_argument('--dumptree', action='store_true')
    parser.add_argument('--dumpsample', action='store_true')
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--weights_vec', type=str, default='')
    parser.add_argument('--evalloss', type=str, choices=['mse', 'msesum', 'dense', 'weightedmse', 'weightedmse++', 'weightedce', 'weightedmsesum', 'dfl', 'quad', 'quad++', 'ce'], default='mse', help='use for evaluate the model with metrics')
    parser.add_argument('--spgrid', type=str, default='(5, 5)', help='the grid of the shortest path')
    parser.add_argument('--solver', type=str, choices=["scip", "gurobi", "glpk"], default="scip", help="optimization solver to use")
    parser.add_argument('--ng_budget', type=int, default=100, help='budget for never grad search')

    args = parser.parse_args()

    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")

    if args.problem == 'budgetalloc':
        from BudgetAllocation import BudgetAllocation
        problem = BudgetAllocation(num_train_instances = args.instances,
                                   num_test_instances = args.testinstances,
                                   num_targets = args.numtargets,
                                   num_items = args.numitems,
                                   budget = args.budget,
                                   num_fake_targets = args.fakefeatures,
                                   rand_seed = args.seed,
                                   val_frac = args.valfrac)
    elif args.problem == 'cubic':
        from CubicTopK import CubicTopK
        problem = CubicTopK(num_train_instances = args.instances,
                            num_test_instances = args.testinstances,
                            num_items = args.numitems,
                            budget = args.budget,
                            rand_seed = args.seed,
                            val_frac = args.valfrac)
    elif args.problem == 'bipartitematching':
        from BipartiteMatching import BipartiteMatching
        problem = BipartiteMatching(num_train_instances = args.instances,
                                    num_test_instances = args.testinstances,
                                    num_nodes = args.nodes,
                                    val_frac = args.valfrac,
                                    rand_seed = args.seed)
    elif args.problem == 'rmab':
        from RMAB import RMAB
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
        from PortfolioOpt import PortfolioOpt
        problem = PortfolioOpt(num_train_instances = args.instances,
                               num_test_instances = args.testinstances,
                               num_stocks = args.stocks,
                               alpha = args.stockalpha,
                               val_frac = args.valfrac,
                               rand_seed = args.seed)
    elif args.problem == 'vmscheduling':
        from VMScheduling import VMScheduling
        problem = VMScheduling(rand_seed=args.seed,
                               num_train=args.instances - int(args.valfrac * args.instances),
                               num_eval=int(args.valfrac * args.instances),
                               num_test=args.testinstances,
                               num_per_instance=args.itempertrace)
    elif args.problem == 'vmschedulingseq':
        from VMScheduling_seq import VMSchedulingSeq
        problem = VMSchedulingSeq(rand_seed=args.seed,
                               num_train=args.instances - int(args.valfrac * args.instances),
                               num_eval=int(args.valfrac * args.instances),
                               num_test=args.testinstances,
                               num_per_instance=args.itempertrace)
    elif args.problem == 'shortestpath':
        from ShortestPath import ShortestPath
        problem = ShortestPath(num_feats=args.numfeatures,
                               grid=eval(args.spgrid),
                               num_train=args.instances - int(args.valfrac * args.instances),
                               num_val=int(args.valfrac * args.instances),
                               num_test=args.testinstances,
                               seed=args.seed,
                               solver=args.solver)



    # Load an ML model to predict the parameters of the problem
    print(f"Building {args.model} Model...")
    if args.model == "xgb_decoupled":
        from train_xgb import train_xgb
        model, metrics = train_xgb(args, problem)
    elif args.model.startswith("xgb_lodl") or args.model.startswith("xgb_coupled"):
        from train_xgb import train_xgb_lodl
        model, metrics = train_xgb_lodl(args, problem)
    elif args.model.startswith("dense"):
        from train_dense import train_dense
        model, metrics = train_dense(args, problem)
    elif args.model.startswith("xgb_search"):
        from train_xgb import train_xgb_search_weights
        model, metrics = train_xgb_search_weights(args, problem)
    elif args.model == "xgb_ngopt":
        from train_xgb import train_xgb_ngopt
        model, metrics = train_xgb_ngopt(args, problem)

    perf_metrics(args, problem, metrics)

    print(args)


