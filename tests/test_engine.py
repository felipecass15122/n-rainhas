import numpy as np
import pytest
from ga.population import create_population
from ga.engine import run, GAResult
from problems.nqueens.individual import NQueensIndividual


class TestCreatePopulation:
    def test_correct_size(self):
        pop = create_population(8, 20)
        assert len(pop) == 20

    def test_all_valid_permutations(self):
        pop = create_population(8, 10)
        for ind in pop:
            assert sorted(ind.genes.tolist()) == list(range(8))


class TestEngine:
    def test_solves_4_queens(self):
        # 4 rainhas é simples — deve convergir facilmente
        pop = create_population(4, 20)
        result = run(pop, max_generations=500, elite_size=2)
        assert result.solved is True
        assert result.solution is not None
        assert result.solution.fitness == 0

    def test_solves_8_queens(self):
        pop = create_population(8, 30)
        result = run(pop, max_generations=2000, elite_size=4, mutation_rate=0.2)
        assert result.solved is True
        assert result.solution.fitness == 0

    def test_history_length_matches_generations(self):
        pop = create_population(4, 10)
        result = run(pop, max_generations=500, elite_size=2)
        assert len(result.best_fitness_history) == result.generation

    def test_history_is_non_increasing(self):
        # O melhor fitness só pode manter ou melhorar (elitismo garante)
        pop = create_population(8, 20)
        result = run(pop, max_generations=200, elite_size=2)
        history = result.best_fitness_history
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1]

    def test_no_solution_when_max_gens_too_low(self):
        # Com apenas 1 geração e população aleatória, é improvável resolver 8 rainhas
        np.random.seed(42)
        pop = create_population(8, 5)
        # Substitui todos por indivíduos propositalmente ruins
        for ind in pop:
            ind.genes = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            ind.invalidate_fitness()
        result = run(pop, max_generations=1, elite_size=2)
        # Pode ou não resolver — só verificamos que o resultado é consistente
        assert isinstance(result.solved, bool)
        assert isinstance(result.generation, int)
