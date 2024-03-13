# Based on https://automl.github.io/SMAC3/main/examples/1_basics/1_quadratic_function.html#sphx-glr-examples-1-basics-1-quadratic-function-py
# Quadratic function
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class QuadraticFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        w1 = Float("w1", (1, 100), default=1)
        w2 = Float("w2", (1, 100), default=1)
        w3 = Float("w3", (1, 100), default=1)

        cs.add_hyperparameters([w1, w2, w3])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        w1 = config["w1"]
        w2 = config["w2"]
        w3 = config["w3"]
        return 2 * w1 * w2 - w1 ** 2 + 2 * w1 * w2 - (w2 ** 2 + w3 ** 2)


if __name__ == "__main__":
    model = QuadraticFunction()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, deterministic=True, n_trials=100)

    # Now we use SMAC to find the best hyperparameters
    smac = HPOFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )

    incumbent = smac.optimize()
    print(f"Get results: {incumbent}")

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")


