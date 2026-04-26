import numpy as np


def count_conflicts(genes: np.ndarray) -> int:
    "Conta o número de pares de rainhas em conflito diagonal."
    n = len(genes)
    conflicts = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(int(genes[i]) - int(genes[j])) == abs(i - j):
                conflicts += 1
    return conflicts
