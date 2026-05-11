import numpy as np
from ga.individual import Individual
from problems.rastrigin.fitness import rastrigin


class RastriginInd(Individual):
    def __init__(self, n_dims: int, genes: np.ndarray | None = None, bounds: np.ndarray | None = None):
        if bounds is None:
            bounds = np.full((n_dims, 2), [-5.12, 5.12])
        if genes is None:
            genes = np.random.uniform(bounds[:, 0], bounds[:, 1])
        super().__init__(genes)
        self.bounds = bounds

    def _evaluate(self) -> float:
        return rastrigin(self.genes)

    def copy(self) -> "RastriginInd":
        return RastriginInd(len(self.genes), genes=self.genes.copy(), bounds=self.bounds.copy())
