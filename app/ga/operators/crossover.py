import numpy as np
from ga.individual import Individual


def single_point_crossover(parent_a: Individual, parent_b: Individual) -> tuple[Individual, Individual]:
    """Crossover de ponto único entre dois pais, com reparação de permutação.
    Retorna dois filhos como cópias do tipo do pai A.
    """
    n = len(parent_a.genes)
    point = np.random.randint(1, n)  # ponto de corte entre [1, n-1]

    child_a_genes = np.concatenate([parent_a.genes[:point], parent_b.genes[point:]])
    child_b_genes = np.concatenate([parent_b.genes[:point], parent_a.genes[point:]])

    child_a_genes = _repair(child_a_genes)
    child_b_genes = _repair(child_b_genes)

    child_a = parent_a.copy()
    child_a.genes = child_a_genes
    child_a.invalidate_fitness()

    child_b = parent_a.copy()
    child_b.genes = child_b_genes
    child_b.invalidate_fitness()

    return child_a, child_b


def _repair(genes: np.ndarray) -> np.ndarray:
    """
    Identifica os valores duplicados e os substitui pelos ausentes,
    preservando a primeira ocorrência de cada valor.
    """
    n = len(genes)
    seen = set()
    duplicates = []  # índices onde há duplicata
    missing = []     # valores que estão faltando

    for i, gene in enumerate(genes):
        if gene in seen:
            duplicates.append(i)
        else:
            seen.add(gene)

    for val in range(n):
        if val not in seen:
            missing.append(val)

    result = genes.copy()
    for idx, val in zip(duplicates, missing):
        result[idx] = val

    return result
