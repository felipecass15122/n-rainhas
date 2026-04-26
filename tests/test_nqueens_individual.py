import numpy as np
import pytest
from problems.nqueens.individual import NQueensIndividual
from problems.nqueens.fitness import count_conflicts


class TestCountConflicts:
    def test_known_solution_has_zero_conflicts(self):
        # Solução válida conhecida para 8 rainhas
        genes = np.array([0, 4, 7, 5, 2, 6, 1, 3])
        assert count_conflicts(genes) == 0

    def test_diagonal_main_has_max_conflicts(self):
        # Todas na diagonal principal: cada par consecutivo conflita
        genes = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        # Pares em conflito: (0,1),(0,2),...,(6,7) — todos os C(8,2)=28 pares
        assert count_conflicts(genes) == 28

    def test_reverse_diagonal_also_conflicts(self):
        genes = np.array([7, 6, 5, 4, 3, 2, 1, 0])
        assert count_conflicts(genes) == 28

    def test_single_queen_no_conflict(self):
        assert count_conflicts(np.array([0])) == 0

    def test_two_queens_no_conflict(self):
        # Rainha em (0,0) e (1,2): |0-2| != |0-1|
        assert count_conflicts(np.array([0, 2])) == 0

    def test_two_queens_in_conflict(self):
        # Rainha em (0,0) e (1,1): |0-1| == |0-1|
        assert count_conflicts(np.array([0, 1])) == 1


class TestNQueensIndividual:
    def test_random_individual_has_valid_permutation(self):
        ind = NQueensIndividual(8)
        assert sorted(ind.genes.tolist()) == list(range(8))

    def test_fitness_cached(self):
        ind = NQueensIndividual(8, np.array([0, 4, 7, 5, 2, 6, 1, 3]))
        f1 = ind.fitness
        f2 = ind.fitness
        assert f1 == f2 == 0

    def test_solution_fitness_is_zero(self):
        ind = NQueensIndividual(8, np.array([0, 4, 7, 5, 2, 6, 1, 3]))
        assert ind.fitness == 0

    def test_copy_is_independent(self):
        ind = NQueensIndividual(8, np.array([0, 4, 7, 5, 2, 6, 1, 3]))
        clone = ind.copy()
        clone.genes[0] = 99
        assert ind.genes[0] == 0  # original não foi modificado

    def test_copy_preserves_genes(self):
        ind = NQueensIndividual(8, np.array([0, 4, 7, 5, 2, 6, 1, 3]))
        clone = ind.copy()
        assert clone.genes.tolist() == ind.genes.tolist()

    def test_invalidate_fitness(self):
        ind = NQueensIndividual(8, np.array([0, 4, 7, 5, 2, 6, 1, 3]))
        _ = ind.fitness  # popula cache
        ind.genes[0] = 1  # introduz conflito
        ind.invalidate_fitness()
        assert ind.fitness > 0

    def test_4_queens_solution(self):
        # Solução válida para 4 rainhas: [1, 3, 0, 2]
        ind = NQueensIndividual(4, np.array([1, 3, 0, 2]))
        assert ind.fitness == 0
