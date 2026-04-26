import numpy as np
from ga.individual import Individual


def roulette_select(population: list[Individual], n: int) -> list[Individual]:
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)
    # Invertemos: quanto menor o fitness, maior o peso
    weights = 1.0 / (fitnesses + 1.0)
    total = weights.sum()
    probabilities = weights / total

    chosen_indices = np.random.choice(len(population), size=n, replace=True, p=probabilities)
    return [population[i].copy() for i in chosen_indices]
