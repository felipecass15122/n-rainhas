import numpy as np
from ga.individual import Individual


def swap_mutation(individual: Individual) -> Individual:
    "Mutação por troca: permuta duas posições aleatórias dos genes."
    n = len(individual.genes)
    i, j = np.random.choice(n, size=2, replace=False)
    individual.genes[i], individual.genes[j] = individual.genes[j], individual.genes[i]
    individual.invalidate_fitness()
    return individual


def gaussian_mutation(individual: Individual, sigma: float, mutation_rate: float | None = None) -> Individual:
    if mutation_rate is None:
        mutation_rate = 1.0 / len(individual.genes)
    mask = np.random.rand(len(individual.genes)) < mutation_rate
    individual.genes[mask] += np.random.normal(0, sigma, mask.sum())
    if individual.bounds is not None:
        individual.genes = np.clip(
            individual.genes,
            individual.bounds[:, 0],
            individual.bounds[:, 1],
        )
    individual.invalidate_fitness()
    return individual
