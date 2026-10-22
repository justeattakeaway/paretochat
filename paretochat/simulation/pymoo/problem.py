import numpy as np
from pymoo.core.problem import Problem


class KnapsackPymooProblem(Problem):
    _BINARY_SOLUTION_THRESHOLD = 0.5

    def __init__(self, instance_path: str, pareto_front_path: str):
        self._load_instance(instance_path)
        self._load_pareto_front(pareto_front_path)

        super().__init__(
            n_var=self.num_items,
            n_obj=self.num_objectives,
            n_ieq_constr=1,
            xl=np.zeros(self.num_items),
            xu=np.ones(self.num_items),
            vtype=int,
        )

    def _load_instance(self, instance_path: str):
        with open(instance_path, "r") as f:
            lines = f.readlines()

        self.num_items = int(lines[0].strip().split()[0])
        self.num_objectives = int(lines[0].strip().split()[1])
        self.capacity = int(lines[1].strip())

        self.items = []
        for i in range(self.num_items):
            parts = list(map(int, lines[2 + i].strip().split()))
            weight = parts[0]
            obj_values = parts[1 : 1 + self.num_objectives]
            self.items.append((weight, obj_values))

    def _load_pareto_front(self, pareto_front_path: str):
        self.pareto_front = np.loadtxt(pareto_front_path, delimiter=" ")

    def _calc_pareto_front(self):
        # Note: We negate the objective values because pymoo minimizes by default, but we want to maximize the objectives in the knapsack problem
        return -self.pareto_front

    def _calc_pareto_set(self, *args, **kwargs):
        return self._calc_pareto_front()

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a 2D array of shape (n_samples, num_items) where each row is a binary vector indicating which items are included in the knapsack
        n_samples = x.shape[0]
        f = np.zeros((n_samples, self.n_obj))
        g = np.zeros((n_samples, 1))

        for i in range(n_samples):
            total_weight = 0
            for j in range(self.n_var):
                # Use the defined threshold to determine if an item is included
                if x[i, j] > KnapsackPymooProblem._BINARY_SOLUTION_THRESHOLD:
                    total_weight += self.items[j][0]
                    for k in range(self.n_obj):
                        f[i, k] += self.items[j][1][k]
            g[i] = max(0, total_weight - self.capacity)

        # Note: We negate the objective values because pymoo minimizes by default, but we want to maximize the objectives in the knapsack problem
        out["F"] = -f
        out["G"] = g
