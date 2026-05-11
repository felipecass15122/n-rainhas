import numpy as np

DEFAULT_M = 5
DEFAULT_C = np.array([1, 2, 5, 2, 3], dtype=float)
DEFAULT_A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]], dtype=float)


def langermann(genes: np.ndarray, m: int, c: np.ndarray, A: np.ndarray) -> float:
    total = 0.0
    for i in range(m):
        diff = genes - A[i]
        inner = float(np.sum(diff**2))
        total += c[i] * np.exp(-inner / np.pi) * np.cos(np.pi * inner)
    return float(-total)
