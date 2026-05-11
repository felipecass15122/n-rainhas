from problems.nqueens.individual import NQueensIndividual
from app.problems.rastrigin.individual import RastriginInd
from ga.individual import Individual


def create_population(n_queens: int, size: int) -> list[Individual]:
    return [NQueensIndividual(n_queens) for _ in range(size)]


def create_rastrigin_population(n_dims: int, size: int) -> list[Individual]:
    return [RastriginInd(n_dims) for _ in range(size)]
