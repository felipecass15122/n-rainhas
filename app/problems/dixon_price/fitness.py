import numpy as np


def dixon_price(genes: np.ndarray) -> float:
    n = len(genes)
    term1 = (genes[0] - 1.0) ** 2
    i = np.arange(2, n + 1, dtype=float)
    term2 = np.sum(i * (2.0 * genes[1:] ** 2 - genes[:-1]) ** 2)
    return float(term1 + term2)
