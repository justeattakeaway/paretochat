import numpy as np
from pymoo.algorithms.moo.pinsga2 import AutomatedDM


class LinearUtilityPymooDM(AutomatedDM):
    def __init__(self, weights, maximize=False):
        super().__init__()
        self.weights = weights
        self.maximize = maximize

    def makeDecision(self, F):
        # F is a 2 X m matrix of m-objective values for the two solutions being compared
        val = np.dot(F, self.weights)

        if not self.maximize:
            val = -val

        if val[0] > val[1]:
            out = "a"
        elif val[0] < val[1]:
            out = "b"
        else:
            out = "c"

        return out

    def makePairwiseDecision(self, F):
        return super().makePairwiseDecision(F)
