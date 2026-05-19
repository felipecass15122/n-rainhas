import numpy as np
from ga.individual import Individual
from problems.dixon_price.fitness import dixon_price


class DixonPriceInd(Individual):
    def __init__(self, n_dims: int, genes: np.ndarray | None = None, bounds: np.ndarray | None = None):
        if bounds is None:
            bounds = np.full((n_dims, 2), [-10.0, 10.0])
        if genes is None:
            genes = np.random.uniform(bounds[:, 0], bounds[:, 1])
        super().__init__(genes)
        self.bounds = bounds

    def _evaluate(self) -> float:
        return dixon_price(self.genes)

    def copy(self) -> "DixonPriceInd":
        return DixonPriceInd(len(self.genes), genes=self.genes.copy(), bounds=self.bounds.copy())
