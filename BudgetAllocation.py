from PThenO import PThenO
import pickle
import random
import numpy as np
from SubmodularOptimizer import SubmodularOptimizer
import torch


class BudgetAllocation(PThenO):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=80,  # number of instances to use from the dataset to train
        num_val_instances=20,
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_targets=10,  # number of items to choose from
        num_items=5,  # number of targets to consider
        budget=2,  # number of items that can be picked
        num_fake_targets=500  # number of random features added to make the task harder # val_frac=0.2,  # fraction of training data reserved for validation
    ):
        super(BudgetAllocation, self).__init__()

        # Load train and test labels
        self.num_train_instances = num_train_instances
        self.num_val_instances = num_val_instances
        self.num_test_instances = num_test_instances

        self.Ys_train, self.Ys_test = self._load_instances(num_train_instances + num_val_instances, num_test_instances, num_items, num_targets)

        # Generate features based on the labels
        self.num_items = num_items
        self.num_targets = num_targets
        self.num_fake_targets = num_fake_targets
        self.num_features = self.num_targets + self.num_fake_targets
        self.Xs_train, self.Xs_test = self._generate_features([self.Ys_train, self.Ys_test], self.num_fake_targets)  # features
        assert not (torch.isnan(self.Xs_train).any() or torch.isnan(self.Xs_test).any())

        # Split training data into train/val
        indices = list(range(num_train_instances + num_val_instances))
        random.shuffle(indices)

        self.val_idxs = indices[:num_val_instances]
        self.train_idxs = indices[num_val_instances: (num_val_instances + num_train_instances)]
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])

        # Create functions for optimisation
        assert budget < num_items
        self.budget = budget
        self.opt = SubmodularOptimizer(self.get_objective, self.budget)

        self.num_feats = self.num_items * self.num_targets ## TODO check correct?

        # Undo random seed setting
        # self._set_seed()

    def _load_instances(self, num_train, num_test, num_items, num_targets):
        """
        Loads the labels (Ys) of the prediction from a file, and returns a subset of it parameterised by instances.
        """
        # Load the dataset
        with open('data/budget_allocation_data.pkl', 'rb') as f:
            Yfull, _ = pickle.load(f, encoding='bytes')
        Yfull = np.array(Yfull)

        N = len(Yfull)
        indices = list(range(N))
        random.shuffle(indices)

        assert (num_train + num_test) <= N
        traininds = indices[:num_train]
        testinds = indices[num_train:(num_train + num_test)]

        # Whittle the dataset down to the right size
        def whittle(matrix, size, dim):
            assert size <= matrix.shape[dim]
            elements = np.random.choice(matrix.shape[dim], size)
            return np.take(matrix, elements, axis=dim)

        Ytrains = Yfull[traininds]
        Ytrains = whittle(Ytrains, num_items, 1)
        Ytrains = whittle(Ytrains, num_targets, 2)
        Ytrains = torch.from_numpy(Ytrains).float().detach()

        Ytests = Yfull[testinds]
        Ytests = whittle(Ytests, num_items, 1)
        Ytests = whittle(Ytests, num_targets, 2)
        Ytests = torch.from_numpy(Ytests).float().detach()

        assert not torch.isnan(Ytrains).any()
        assert not torch.isnan(Ytests).any()

        return Ytrains, Ytests

    def _generate_features(self, Ysets, num_fake_targets):
        """
        Converts labels (Ys) + random noise, to features (Xs)
        """
        # Generate random matrix common to all Ysets (train + test)
        transform_nn = torch.nn.Sequential(torch.nn.Linear(self.num_features, self.num_targets))

        # Generate training data by scrambling the Ys based on this matrix
        Xsets = []
        for Ys in Ysets:
            # Normalise data across the last dimension
            Ys_mean = Ys.reshape((-1, Ys.shape[2])).mean(dim=0)
            Ys_std = Ys.reshape((-1, Ys.shape[2])).std(dim=0)
            Ys_standardised = (Ys - Ys_mean) / (Ys_std + 1e-10)
            assert not torch.isnan(Ys_standardised).any()

            # Add noise to the data to complicate prediction
            fake_features = torch.normal(mean=torch.zeros(Ys.shape[0], Ys.shape[1], num_fake_targets))
            Ys_augmented = torch.cat((Ys_standardised, fake_features), dim=2)

            # Encode Ys as features by multiplying them with a random matrix
            Xs = transform_nn(Ys_augmented).detach().clone()
            Xsets.append(Xs)

        return (*Xsets,)

    def get_train_data(self):
        return self.Xs_train[self.train_idxs], self.Ys_train[self.train_idxs],  [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_train[self.val_idxs], self.Ys_train[self.val_idxs],  [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test,  [None for _ in range(len(self.Ys_test))]

    def get_modelio_shape(self):
        return self.num_features, self.num_targets
    
    def get_output_activation(self):
        return 'relu'

    def get_twostageloss(self):
        return 'mse'

    def get_objective(self, Y, Z, w=None, **kwargs):
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        # Sanity check inputs
        assert Y.shape[-2] == Z.shape[-1]
        assert len(Z.shape) + 1 == len(Y.shape)

        # Initialise weights to default value
        if w is None:
            w = torch.ones(Y.shape[-1]).requires_grad_(False)
        else:
            assert Y.shape[-1] == w.shape[0]
            assert len(w.shape) == 1

        # Calculate objective
        p_fail = 1 - Z.unsqueeze(-1) * Y
        p_all_fail = p_fail.prod(dim=-2)
        obj = (w * (1 - p_all_fail)).sum(dim=-1)
        return obj

    def get_decision(self, Y, Z_init=None, **kwargs):
        # If this is a single instance of a decision problem
        if len(Y.shape) == 2:
            return self.opt(Y, Z_init=Z_init)

        # If it's not...
        #   Remember the shape
        Y_shape = Y.shape
        #   Break it down into individual instances and solve
        Y_new = Y.view((-1, Y_shape[-2], Y_shape[-1]))
        Z = torch.cat([self.opt(y, Z_init=Z_init) for y in Y_new], dim=0)
        #   Convert it back to the right shape
        Z = Z.view((*Y_shape[:-2], -1))
        return Z

    def dec_loss_3d(self, z_pred: np.ndarray, z_true: np.ndarray, w=None, verbose=False, **kwargs) -> np.ndarray:
        # Function signature is from the https://github.com/facebookresearch/LANCER
        # TODO Need to check how does SubmodularOptimizer work

        Ypred = torch.tensor(z_pred)
        Ytrue = torch.tensor(z_true)

        dec = self.get_decision(Ypred)
        obj = self.get_objective(Ytrue, dec, w=w)
        return -1 * obj.detach().numpy()

    def dec_loss(self, z_pred: np.ndarray, z_true: np.ndarray, **kwargs) -> np.ndarray:
        Y = torch.tensor(z_pred.reshape(len(z_pred), self.num_items, self.num_targets))
        assert len(Y.shape) > 2
        # If it's not...
        #   Remember the shape
        Y_shape = Y.shape
        #   Break it down into individual instances and solve
        Y_new = Y.view((-1, Y_shape[-2], Y_shape[-1]))
        Z = torch.cat([self.opt(y) for y in Y_new], dim=0)
        #   Convert it back to the right shape
        Z = Z.view((*Y_shape[:-2], -1))

        Y_gold = torch.tensor(z_true.reshape(len(z_true), self.num_items, self.num_targets))

        assert Y_gold.shape[-2] == Z.shape[-1]
        assert len(Z.shape) + 1 == len(Y_gold.shape)

        w = torch.ones(Y_gold.shape[-1]).requires_grad_(False)
        # Calculate objective
        p_fail = 1 - Z.unsqueeze(-1) * Y_gold
        p_all_fail = p_fail.prod(dim=-2)
        obj = (w * (1 - p_all_fail)).sum(dim=-1).unsqueeze(dim=-1).detach().numpy()
        return -1 * obj

if __name__ == "__main__":
    random.seed(0) # For debug
    prob = BudgetAllocation()

    Xtrain, Ytrain, _ = prob.get_train_data()

    xtrain2d = Xtrain.cpu().detach().numpy().reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])
    ytrain2d = Ytrain.cpu().detach().numpy().reshape(Ytrain.shape[0], Ytrain.shape[1] * Ytrain.shape[2])





