from problems.nqueens.individual import NQueensIndividual
from ga.individual import Individual


def create_population(n_queens: int, size: int) -> list[Individual]:
    return [NQueensIndividual(n_queens) for _ in range(size)]
