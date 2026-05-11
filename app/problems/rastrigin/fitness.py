import numpy as np

A = 10


def rastrigin(genes: np.ndarray) -> float:
    n = len(genes)
    return float(A * n + np.sum(genes**2 - A * np.cos(2 * np.pi * genes)))
