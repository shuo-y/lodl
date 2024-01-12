from PThenO import PThenO
import numpy as np

# Reference of https://arxiv.org/pdf/2305.16830.pdf
# see also https://arxiv.org/pdf/2305.16830.pdf
# Check github https://github.com/paulgrigas/SmartPredictThenOptimize

class ExampleProb(PThenO):
    def __init__(self):
        super(ExampleProb, self).__init__()
        pass

    def dec_loss(self, z_pred: np.ndarray, z_true: np.ndarray, verbose=False, **kwargs) -> np.ndarray:
        choose_ind = np.argmin(z_pred, axis=1)
        true_ind = np.argmin(z_true, axis=1)
        zpred_dec = np.take_along_axis(z_true, np.expand_dims(choose_ind, axis=1), axis=1)
        zopt_dec = np.take_along_axis(z_true, np.expand_dims(true_ind, axis=1), axis=1)
        diff = (zpred_dec - zopt_dec) ** 2
        diff = diff.sum(axis=1)
        return diff


    def _generate_dataset_single_fun(self, start, end, num, fun):
        x = np.linspace(start, end, num)
        y = np.apply_along_axis(fun, -1, np.expand_dims(x, 1))
        return x, y

    def generate_dataset_two_fun(self, start1, end1, num1, start2, end2, num2, fun1, fun2):
        x1, y1 = self._generate_dataset_single_fun(start1, end1, num1, fun1)
        x2, y2 = self._generate_dataset_single_fun(start2, end2, num2, fun2)
        xy1 = np.concatenate([np.expand_dims(x1, axis=1), y1], axis=1)
        xy2 = np.concatenate([np.expand_dims(x2, axis=1), y2], axis=1)

        xy = np.concatenate([xy1, xy2], axis=0)
        np.random.shuffle(xy)
        return xy[:,0], xy[:, 1:]





    def get_decision(self):
        pass

    def get_modelio_shape(self):
        pass

    def get_objective(self):
        pass

    def get_output_activation(self):
        pass



if __name__ == "__main__":
    def fun1(x):
        return (0, 0.55)

    def fun2(x):
        return (1, 0.55)

    prob = ExampleProb()

    x, y = prob.generate_dataset_two_fun(0, 1, 50, 0, 1, 50, fun1, fun2)








