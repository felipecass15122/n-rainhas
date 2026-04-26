import numpy as np
from ga.individual import Individual


def swap_mutation(individual: Individual) -> Individual:
    "Mutação por troca: permuta duas posições aleatórias dos genes."
    n = len(individual.genes)
    i, j = np.random.choice(n, size=2, replace=False)
    individual.genes[i], individual.genes[j] = individual.genes[j], individual.genes[i]
    individual.invalidate_fitness()
    return individual
