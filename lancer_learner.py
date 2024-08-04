# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# It is from https://github.com/facebookresearch/LANCER used for possible baseline tests

import numpy as np
import abc
import torch
import numpy as np
from torch import nn
from torch import optim
from typing import Union
from PThenO import PThenO


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        # layers.append(nn.Dropout(p=0.6))
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)

class BaseLancer(object, metaclass=abc.ABCMeta):
    def predict(self, z_pred: np.ndarray, z_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_theta_step(self, z_pred_tensor: torch.FloatTensor, z_true_tensor: torch.FloatTensor):
        """
        Used to train a target model in models_c. That is, we minimize the LANCER loss
        :param z_pred_tensor: predicted problem descriptors (in pytorch format)
        :param z_true_tensor: ground truth problem descriptors (in pytorch format)
        :return: LANCER loss
        """
        raise NotImplementedError

    def update(self, z_pred: np.ndarray, z_true: np.ndarray, f_hat: np.ndarray, **kwargs):
        """
        Update parameters of LANCER using (z_pred, z_true) and
        decision loss f_hat (true objective value)
        :param z_pred: predicted problem descriptors
        :param z_true: ground truth problem descriptors
        :param f_hat: decision loss (i.e., true objective value)
        :param kwargs: additional problem specific parameters
        :return: None
        """
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError


class MLPLancer(BaseLancer, nn.Module):
    # multilayer perceptron LANCER Model
    def __init__(self,
                 z_dim,
                 f_dim,
                 n_layers,
                 layer_size,
                 learning_rate,
                 opt_type="adam",  # "adam" or "sgd"
                 momentum=0.9,
                 weight_decay=0.001,
                 out_activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.f_dim = f_dim
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        #####################################
        self.model_output = build_mlp(input_size=self.z_dim,
                                      output_size=self.f_dim,
                                      n_layers=self.n_layers,
                                      size=self.layer_size,
                                      output_activation=out_activation)
        #self.model_output.to(nn_utils.device)
        self.loss = nn.MSELoss()
        if opt_type == "adam":
            self.optimizer = optim.Adam(params=self.model_output.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.SGD(params=self.model_output.parameters(),
                                       lr=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def predict(self, z_pred: np.ndarray, z_true: np.ndarray) -> np.ndarray:
        assert z_pred.shape == z_true.shape
        self.mode(train=False)
        with torch.no_grad():
            if len(z_true.shape) > 1:
                z_pred_tensor, z_true_tensor = torch.from_numpy(z_pred).float(), torch.from_numpy(z_true).float()
            else:
                z_pred_tensor, z_true_tensor = torch.from_numpy(z_pred[None]).float(), torch.from_numpy(z_true[None]).float()
            # return the output of the parametric loss function in numpy format
            return self.forward(z_pred_tensor, z_true_tensor).to("cpu").detach().numpy()

    def mode(self, train=True):
        if train:
            self.model_output.train()
        else:
            self.model_output.eval()

    def forward(self, z_pred_tensor: torch.FloatTensor, z_true_tensor: torch.FloatTensor):
        # input = torch.cat((z_pred_tensor, z_true_tensor), dim=1)
        # input = torch.abs(z_true_tensor - z_pred_tensor)
        input = torch.square(z_true_tensor - z_pred_tensor)
        return self.model_output(input)

    def forward_theta_step(self, z_pred_tensor: torch.FloatTensor, z_true_tensor: torch.FloatTensor):
        predicted_loss = self.forward(z_pred_tensor, z_true_tensor)
        return torch.mean(predicted_loss)

    def update(self, z_pred: np.ndarray, z_true: np.ndarray, f_hat: np.ndarray, **kwargs):
        z_pred_tensor = torch.from_numpy(z_pred).float()
        z_true_tensor = torch.from_numpy(z_true).float()  # fixed input
        f_hat_tensor = torch.from_numpy(f_hat).float() # targets
        predictions = self.forward(z_pred_tensor, z_true_tensor)
        self.optimizer.zero_grad()
        loss = self.loss(predictions, f_hat_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class BaseCModel(object, metaclass=abc.ABCMeta):
    def predict(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def initial_fit(self, y: np.ndarray, z_true: np.ndarray, **kwargs):
        """Initialize the model by fitting it to the ground truth z_true"""
        raise NotImplementedError

    def update(self, y: np.ndarray, z_true: np.ndarray, model_loss: BaseLancer):
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError


class MLPCModel(BaseCModel, nn.Module):
    # multilayer perceptron CModel
    def __init__(self,
                 y_dim,
                 z_dim,
                 n_layers,
                 layer_size,
                 learning_rate,
                 opt_type="adam", # "adam" or "sgd"
                 momentum=0.9,
                 weight_decay=0.001,
                 z_regul=0.0,
                 activation="tanh",
                 output_activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.z_regul = z_regul
        #####################################
        self.model_output = build_mlp(input_size=self.y_dim,
                                      output_size=self.z_dim,
                                      n_layers=self.n_layers,
                                      size=self.layer_size,
                                      activation=activation,
                                      output_activation=output_activation)
        #self.model_output.to(nn_utils.device)
        self.loss = nn.MSELoss()
        if opt_type == "adam":
            self.optimizer = optim.Adam(params=self.model_output.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.SGD(params=self.model_output.parameters(),
                                       lr=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def initial_fit(self, y: np.ndarray, z_true: np.ndarray,
                    learning_rate=0.005, num_epochs=100, batch_size=64, print_freq=1):
        assert y.shape[0] == z_true.shape[0]
        N = y.shape[0]
        n_batches = int(N / batch_size)

        optimizer = optim.Adam(params=self.model_output.parameters(),
                               lr=learning_rate,
                               weight_decay=self.weight_decay)
        self.mode(train=True)
        for itr in range(num_epochs):
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                y_batch = torch.from_numpy(y[idxs]).float()
                z_true_batch = torch.from_numpy(z_true[idxs]).float()
                z_pred_batch = self.forward(y_batch)
                optimizer.zero_grad()
                loss = self.loss(z_pred_batch, z_true_batch)
                loss.backward()
                optimizer.step()
            if (itr + 1) % print_freq == 0:
                print("*** Initial fit epoch: ", itr, ", loss: ", loss.item())

    def predict(self, y: np.ndarray) -> np.ndarray:
        self.mode(train=False)
        with torch.no_grad():
            if len(y.shape) > 1:
                y_tensor = torch.from_numpy(y).float()
            else:
                y_tensor = torch.from_numpy(y[None]).float()
            z_pred_tensor = self.forward(y_tensor)
            return z_pred_tensor.to("cpu").detach().numpy()

    def mode(self, train=True):
        if train:
            self.model_output.train()
        else:
            self.model_output.eval()

    def forward(self, y_tensor: torch.FloatTensor):
        return torch.squeeze(self.model_output(y_tensor))

    def update(self, y: np.ndarray, z_true: np.ndarray, model_loss: BaseLancer):
        y_tensor = torch.from_numpy(y).float()
        z_true_tensor = torch.from_numpy(z_true).float()
        z_pred_tensor = self.forward(y_tensor)
        self.optimizer.zero_grad()
        lancer_loss = model_loss.forward_theta_step(z_pred_tensor, z_true_tensor)
        total_loss = lancer_loss + self.z_regul * self.loss(z_pred_tensor, z_true_tensor)
        total_loss.backward()
        self.optimizer.step()
        return lancer_loss.item()


class LancerLearner:
    def __init__(self, params: dict, c_model_type: str, lancer_model_type: str, bb_problem: PThenO):
        self.bb_problem = bb_problem
        self.y_dim = self.bb_problem.num_feats
        #self.c_out_dim, self.lancer_in_dim = self.bb_problem.get_c_shapes() # What does this mean??
        self.f_dim = 1
        #c_hidden_activation, c_output_activation = self.bb_problem.get_activations()
        if lancer_model_type == "mlp":
            self.lancer_model = MLPLancer(params["lancer_in_dim"], self.f_dim,
                                          n_layers=params["lancer_n_layers"],
                                          layer_size=params["lancer_layer_size"],
                                          learning_rate=params["lancer_lr"],
                                          opt_type=params["lancer_opt_type"],
                                          weight_decay=params["lancer_weight_decay"],
                                          out_activation=params["lancer_out_activation"])
        else:
            raise NotImplementedError
        if c_model_type == "mlp":
            self.cmodel = MLPCModel(self.y_dim, params["c_out_dim"],
                                    n_layers=params["c_n_layers"],
                                    layer_size=params["c_layer_size"],
                                    learning_rate=params["c_lr"],
                                    opt_type=params["c_opt_type"],
                                    weight_decay=params["c_weight_decay"],
                                    z_regul=params["z_regul"],
                                    activation=params["c_hidden_activation"],
                                    output_activation=params["c_output_activation"])
        else:
            raise NotImplementedError

    def learn_theta(self, y, z_true, max_iter=1, batch_size=64, print_freq=1):
        """
        Fitting target model C_{\theta}
        """
        assert y.shape[0] == z_true.shape[0]
        N = y.shape[0]
        n_batches = int(N / batch_size)
        self.lancer_model.mode(train=False)
        self.cmodel.mode(train=True)
        total_iter = 0
        while total_iter < max_iter:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                z_true_batch = z_true[idxs]
                y_batch = y[idxs]
                loss_i = self.cmodel.update(y_batch, z_true_batch, self.lancer_model)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting target model C, itr: ", total_iter, ", lancer loss: ", loss_i, flush=True)
                if total_iter >= max_iter:
                    break

    def learn_w(self, z_pred, z_true, f_hat, max_iter=1, batch_size=64, print_freq=1):
        """
        Fitting LANCER model
        """
        assert z_pred.shape == z_true.shape
        assert z_true.shape[0] == f_hat.shape[0]
        N = z_true.shape[0]
        n_batches = int(N / batch_size)
        self.lancer_model.mode(train=True)
        total_iter = 0
        while total_iter < max_iter:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                f_hat_batch = f_hat[idxs]
                z_true_batch = z_true[idxs]
                z_pred_batch = z_pred[idxs]
                loss_i = self.lancer_model.update(z_pred_batch, z_true_batch, f_hat_batch)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting lancer, itr: ", total_iter, ", loss: ", loss_i, flush=True)
                if total_iter >= max_iter:
                    break

    def run_training_loop(self, dataset, n_iter, **kwargs):
        c_max_iter = kwargs["c_max_iter"] if "c_max_iter" in kwargs else 100
        c_nbatch = kwargs["c_nbatch"] if "c_nbatch" in kwargs else 128
        c_epochs_init = kwargs["c_epochs_init"] if "c_epochs_init" in kwargs else 30
        c_lr_init = kwargs["c_lr_init"] if "c_lr_init" in kwargs else 0.005
        lancer_max_iter = kwargs["lancer_max_iter"] if "lancer_max_iter" in kwargs else 100
        lancer_nbatch = kwargs["lancer_nbatch"] if "lancer_nbatch" in kwargs else 1000
        print_freq = kwargs["print_freq"] if "print_freq" in kwargs else 1
        use_replay_buffer = kwargs["use_replay_buffer"] if "use_replay_buffer" in kwargs else False

        Y_train, Y_test, Z_train, Z_test, Z_train_aux, Z_test_aux = dataset
        self.cmodel.initial_fit(Y_train, Z_train, c_lr_init,
                                num_epochs=c_epochs_init,
                                batch_size=c_nbatch,
                                print_freq=print_freq)

        for itr in range(n_iter):
            #print("\n ---- Running solver for train set -----")
            Z_pred = self.cmodel.predict(Y_train)
            f_hat = self.bb_problem.dec_loss(Z_pred, Z_train, aux_data=Z_train_aux)

            # if true, adding model evaluations to the "replay buffer"
            if itr == 0 or not use_replay_buffer:
                Z_pred_4lancer = np.array(Z_pred)
                Z_true_4lancer = np.array(Z_train)
                f_hat_4lancer = np.array(f_hat)
            else:
                Z_pred_4lancer = np.vstack((Z_pred_4lancer, Z_pred))
                Z_true_4lancer = np.vstack((Z_true_4lancer, Z_train))
                f_hat_4lancer = np.vstack((f_hat_4lancer, f_hat))

            # Step over w: learning LANCER
            self.learn_w(Z_pred_4lancer, Z_true_4lancer, f_hat_4lancer,
                         max_iter=lancer_max_iter, batch_size=lancer_nbatch, print_freq=print_freq)

            # Step over theta: learning target model c
            self.learn_theta(Y_train, Z_train,
                             max_iter=c_max_iter, batch_size=c_nbatch, print_freq=print_freq)

        return self.cmodel


def test_lancer(prob, xtrain, ytrain, auxtrain, xtest, ytest, auxtest, lancer_in_dim,
                c_out_dim, n_iter, c_max_iter, c_nbatch, lancer_max_iter, lancer_nbatch,
                c_epochs_init, c_lr_init, lancer_lr=0.001, c_lr=0.005,
                lancer_n_layers=2, lancer_layer_size=100, c_n_layers=0, c_layer_size=64,
                lancer_weight_decay=0.01, c_weight_decay=0.01, z_regul=0.0,
                lancer_out_activation="relu", c_hidden_activation="tanh", c_output_activation="relu", print_freq=1):
    # This default params is based on the original LANCER paper
    # See also https://arxiv.org/pdf/2307.08964
    def_param = {"lancer_in_dim": lancer_in_dim,
                 "c_out_dim": c_out_dim,
                 "lancer_n_layers": 2,
                 "lancer_layer_size": 100,
                 "lancer_lr": lancer_lr,
                 "lancer_opt_type": "adam",
                 "lancer_weight_decay": lancer_weight_decay,
                 "c_n_layers": c_n_layers,
                 "c_layer_size": c_layer_size,
                 "c_lr": c_lr,
                 "c_opt_type": "adam",
                 "c_weight_decay": c_weight_decay,
                 "z_regul": z_regul,
                 "lancer_out_activation": lancer_out_activation,
                 "c_hidden_activation": c_hidden_activation,
                 "c_output_activation": c_output_activation}

    learner = LancerLearner(def_param, "mlp", "mlp", prob)
    dataset = (xtrain, None, ytrain, None, auxtrain, None)
    lancer_model = learner.run_training_loop(dataset,
                                             n_iter=n_iter,
                                             c_max_iter=c_max_iter,
                                             c_nbatch=c_nbatch,
                                             lancer_max_iter=lancer_max_iter,
                                             lancer_nbatch=lancer_nbatch,
                                             c_epochs_init=c_epochs_init,
                                             c_lr_init=c_lr_init,
                                             print_freq=print_freq)

    ytestpred = lancer_model.predict(xtest)
    testdl = prob.dec_loss(ytestpred, ytest, aux_data=auxtest).flatten()

    ytrainpred = lancer_model.predict(xtrain)
    traindl = prob.dec_loss(ytrainpred, ytrain, aux_data=auxtrain).flatten()

    return lancer_model, traindl, testdl

