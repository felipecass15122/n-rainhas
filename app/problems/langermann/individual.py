import numpy as np
from ga.individual import Individual
from problems.langermann.fitness import langermann, DEFAULT_M, DEFAULT_C, DEFAULT_A


class LangermannInd(Individual):
    def __init__(
        self,
        n_dims: int,
        genes: np.ndarray | None = None,
        bounds: np.ndarray | None = None,
        m: int = DEFAULT_M,
        c: np.ndarray = DEFAULT_C,
        A: np.ndarray | None = None,
    ):
        if bounds is None:
            bounds = np.full((n_dims, 2), [0.0, 10.0])
        if A is None:
            if n_dims == 2:
                A = DEFAULT_A.copy()
            else:
                A = np.random.uniform(0, 10, size=(m, n_dims))
        if genes is None:
            genes = np.random.uniform(bounds[:, 0], bounds[:, 1])
        super().__init__(genes)
        self.bounds = bounds
        self.m = m
        self.c = c
        self.A = A

    def _evaluate(self) -> float:
        return langermann(self.genes, self.m, self.c, self.A)

    def copy(self) -> "LangermannInd":
        return LangermannInd(
            len(self.genes),
            genes=self.genes.copy(),
            bounds=self.bounds.copy(),
            m=self.m,
            c=self.c.copy(),
            A=self.A.copy(),
        )
