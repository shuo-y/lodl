from copy import deepcopy
import random
import torch
from models import model_dict
from losses import MSE, get_loss_fn
from utils import print_metrics, move_to_gpu

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
        samples_filename_read=args.samples_read,
        input_args=args
    )
    # Get data
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()

    ipdim, opdim = problem.get_modelio_shape()
    model_builder = model_dict[args.model]
    model = model_builder(
        num_features = X_train.shape[1:] if args.model == "dense_coupled" else ipdim,
        num_targets = Y_train.shape[1:] if args.model == "dense_coupled"  else opdim,
        num_layers = args.layers,
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
            metrics = print_metrics(model, problem, loss_fn, f"Iter {iter_idx}", isTrain=True)

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
    metrics = print_metrics(model, problem, loss_fn if args.loss == args.evalloss else get_loss_fn(args.evalloss, problem), "Final_seed{}".format(args.seed))

    return model, metrics


def nn2st_iter(problem, xtrain, ytrain, xval, yval, lr, iters, batchsize, n_layers, int_size, earlystopping=False, patience=100, model_type="dense", val_freq=10):
    # Only for Neural Network based two-stage
    # model_type is dense or dense_coupled
    X_train = torch.tensor(xtrain).float()
    Y_train = torch.tensor(ytrain).float()
    if xval is not None and yval is not None:
        X_val = torch.tensor(xval).float()
        Y_val = torch.tensor(yval).float()

    ipdim, opdim = problem.get_modelio_shape()
    model_builder = model_dict[model_type]
    model = model_builder(
        num_features = X_train.shape[1:] if model_type == "dense_coupled" else ipdim,
        num_targets = Y_train.shape[1:] if model_type == "dense_coupled"  else opdim,
        num_layers = n_layers,
        intermediate_size=int_size,
        output_activation=problem.get_output_activation(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best = (float("inf"), None)
    time_since_best = 0
    for iter_idx in range(iters):
        # Check metrics on val set
        if xval is not None and iter_idx % val_freq == 0:
            # Compute metrics
            #metrics = print_metrics(model, problem, loss_fn, f"Iter {iter_idx}", isTrain=True)
            # Get loss on val set
            val_losses = []
            for i in range(len(X_val)):
                pred = model(X_val[i][None]).squeeze()
                val_losses.append(MSE(pred, Y_val[i]))
            # Save model if it's the best one
            val_loss = torch.stack(val_losses).flatten().mean().item()
            if best[1] is None or val_loss < best[0]:
                best = (val_loss, deepcopy(model))
                time_since_best = 0

            # Stop if model hasn't improved for patience steps
            if earlystopping and time_since_best > patience:
                break

        # Learn
        losses = []
        for i in random.sample(range(len(X_train)), min(batchsize, len(X_train))):
            pred = model(X_train[i]).squeeze()
            losses.append(MSE(pred, Y_train[i]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_since_best += 1

    if xval is not None and earlystopping:
        model = best[1]

    # Just return model
    return model



def perf_nn(prob, model, xtest, ytest, auxtest):
    X_test = torch.tensor(xtest).float()
    Y_pred = model(X_test)
    ypred = Y_pred.detach().numpy()
    nntestdl = prob.dec_loss(ypred, ytest, aux_data=auxtest).flatten()
    return nntestdl
