import numpy as np


def print_board(genes: np.ndarray) -> None:
    n = len(genes)
    separator = "+" + ("---+" * n)

    print(separator)
    for row in range(n):
        queen_col = genes[row]
        cells = []
        for col in range(n):
            cells.append(" Q " if col == queen_col else " . ")
        print("|" + "|".join(cells) + "|")
        print(separator)
