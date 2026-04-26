import numpy as np
import pytest
from problems.nqueens.individual import NQueensIndividual
from ga.operators.selection import roulette_select
from ga.operators.crossover import single_point_crossover, _repair
from ga.operators.mutation import swap_mutation


def make_ind(genes: list[int]) -> NQueensIndividual:
    return NQueensIndividual(len(genes), np.array(genes))


# ---------------------------------------------------------------------------
# Seleção por roleta
# ---------------------------------------------------------------------------
class TestRouletteSelect:
    def test_returns_correct_count(self):
        pop = [make_ind([0, 1, 2, 3]), make_ind([1, 3, 0, 2]), make_ind([2, 0, 3, 1])]
        selected = roulette_select(pop, 5)
        assert len(selected) == 5

    def test_selected_are_valid_permutations(self):
        pop = [NQueensIndividual(8) for _ in range(10)]
        selected = roulette_select(pop, 4)
        for ind in selected:
            assert sorted(ind.genes.tolist()) == list(range(8))

    def test_returns_copies_not_references(self):
        pop = [make_ind([0, 1, 2, 3])]
        selected = roulette_select(pop, 1)
        selected[0].genes[0] = 99
        assert pop[0].genes[0] == 0  # original intacto

    def test_best_individual_selected_more_often(self):
        # Indivíduo com fitness 0 deve ser selecionado com maior frequência
        solution = make_ind([1, 3, 0, 2])   # fitness=0
        bad = make_ind([0, 1, 2, 3])         # fitness alto
        pop = [solution, bad]
        selected = roulette_select(pop, 200)
        solution_count = sum(1 for ind in selected if ind.fitness == 0)
        assert solution_count > 100  # bem mais da metade


# ---------------------------------------------------------------------------
# Reparação de permutação
# ---------------------------------------------------------------------------
class TestRepair:
    def test_no_duplicates_unchanged(self):
        genes = np.array([0, 1, 2, 3])
        result = _repair(genes)
        assert result.tolist() == [0, 1, 2, 3]

    def test_single_duplicate_repaired(self):
        # [0, 0, 2, 3] → deve substituir o segundo 0 por 1
        genes = np.array([0, 0, 2, 3])
        result = _repair(genes)
        assert sorted(result.tolist()) == [0, 1, 2, 3]

    def test_result_is_valid_permutation(self):
        genes = np.array([0, 0, 0, 0])
        result = _repair(genes)
        assert sorted(result.tolist()) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Crossover de ponto único
# ---------------------------------------------------------------------------
class TestSinglePointCrossover:
    def test_children_are_valid_permutations(self):
        a = make_ind([0, 1, 2, 3, 4, 5, 6, 7])
        b = make_ind([7, 6, 5, 4, 3, 2, 1, 0])
        for _ in range(20):
            c1, c2 = single_point_crossover(a, b)
            assert sorted(c1.genes.tolist()) == list(range(8))
            assert sorted(c2.genes.tolist()) == list(range(8))

    def test_parents_not_modified(self):
        a = make_ind([0, 1, 2, 3])
        b = make_ind([3, 2, 1, 0])
        orig_a = a.genes.tolist()
        orig_b = b.genes.tolist()
        single_point_crossover(a, b)
        assert a.genes.tolist() == orig_a
        assert b.genes.tolist() == orig_b

    def test_children_fitness_is_calculated(self):
        a = make_ind([0, 1, 2, 3])
        b = make_ind([3, 2, 1, 0])
        c1, c2 = single_point_crossover(a, b)
        assert isinstance(c1.fitness, int)
        assert isinstance(c2.fitness, int)


# ---------------------------------------------------------------------------
# Mutação por swap
# ---------------------------------------------------------------------------
class TestSwapMutation:
    def test_result_is_valid_permutation(self):
        ind = make_ind([0, 1, 2, 3, 4, 5, 6, 7])
        for _ in range(20):
            swap_mutation(ind)
            assert sorted(ind.genes.tolist()) == list(range(8))

    def test_fitness_invalidated_after_mutation(self):
        ind = make_ind([1, 3, 0, 2])
        _ = ind.fitness  # popula cache
        swap_mutation(ind)
        assert ind._fitness is None  # cache foi limpo

    def test_returns_same_individual(self):
        ind = make_ind([0, 1, 2, 3])
        result = swap_mutation(ind)
        assert result is ind
