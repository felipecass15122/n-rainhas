import numpy as np
from ga.individual import Individual


def roulette_select(population: list[Individual], n: int) -> list[Individual]:
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)
    # Deslocamos para que o pior indivíduo tenha peso mínimo positivo,
    # independente de fitness negativos (funciona para N-Queens e funções contínuas)
    worst = fitnesses.max()
    weights = worst - fitnesses + 1.0  # menor fitness → maior peso
    probabilities = weights / weights.sum()

    chosen_indices = np.random.choice(len(population), size=n, replace=True, p=probabilities)
    return [population[i].copy() for i in chosen_indices]
