from abc import ABC, abstractmethod
import numpy as np


class Individual(ABC):
    """Classe base abstrata para indivíduos do algoritmo genético."""

    def __init__(self, genes: np.ndarray):
        self.genes = genes
        self._fitness: int | None = None

    @property
    def fitness(self) -> int:
        if self._fitness is None:
            self._fitness = self._evaluate()
        return self._fitness

    def invalidate_fitness(self):
        """Invalida o cache do fitness após modificação dos genes."""
        self._fitness = None

    @abstractmethod
    def _evaluate(self) -> int:
        "Calcula e retorna o valor de aptidão do indivíduo."

    @abstractmethod
    def copy(self) -> "Individual":
        "Retorna uma cópia independente deste indivíduo."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(genes={self.genes}, fitness={self.fitness})"
