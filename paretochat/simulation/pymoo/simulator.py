from pymoo.algorithms.moo.pinsga2 import PINSGA2
from pymoo.core.result import Result
from pymoo.optimize import minimize

from paretochat.simulation.pymoo.dm import LinearUtilityPymooDM
from paretochat.simulation.pymoo.problem import KnapsackPymooProblem


class PINSGA2Simulator:
    def run(self):
        problem = KnapsackPymooProblem(
            instance_path="paretochat/simulation/instances/knapsack_3_100.txt",
            pareto_front_path="paretochat/simulation/instances/knapsack_3_100_pf.txt",
        )

        dm = LinearUtilityPymooDM(weights=[1.0, 1.0, 1.0])

        algorithm = PINSGA2(vf_type="linear", automated_dm=dm)

        res: Result = minimize(problem, algorithm, ("n_gen", 250), seed=1, verbose=True)

        print("Best solution found: \nF = %s" % (res.F))


if __name__ == "__main__":
    simulator = PINSGA2Simulator()
    simulator.run()
