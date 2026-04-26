import numpy as np
from ga.individual import Individual
from problems.nqueens.fitness import count_conflicts


class NQueensIndividual(Individual):
    "Indivíduo para o problema das N-Rainhas."

    def __init__(self, n: int, genes: np.ndarray | None = None):
        self.n = n
        if genes is None:
            genes = np.random.permutation(n)
        super().__init__(genes)

    def _evaluate(self) -> int:
        return count_conflicts(self.genes)

    def copy(self) -> "NQueensIndividual":
        clone = NQueensIndividual(self.n, self.genes.copy())
        return clone

    def __repr__(self) -> str:
        return f"NQueensIndividual(n={self.n}, genes={self.genes.tolist()}, fitness={self.fitness})"
