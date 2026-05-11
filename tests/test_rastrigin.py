import numpy as np
import pytest
from app.problems.rastrigin.fitness import rastrigin
from app.problems.rastrigin.individual import RastriginInd
from app.ga.population import create_rastrigin_population
from app.ga.operators.crossover import arithmetic_crossover, blx_alpha_crossover
from app.ga.operators.mutation import gaussian_mutation
from app.ga.engine import run


def test_rastrigin_minimum():
    assert rastrigin(np.zeros(10)) == pytest.approx(0.0)


def test_rastrigin_ind_dentro_dos_bounds():
    ind = RastriginInd(n_dims=5)
    assert np.all(ind.genes >= -5.12) and np.all(ind.genes <= 5.12)


def test_rastrigin_copy_e_independente():
    ind = RastriginInd(n_dims=5)
    clone = ind.copy()
    clone.genes[0] = 999.0
    assert ind.genes[0] != 999.0


def test_arithmetic_crossover_bounds():
    pa = RastriginInd(n_dims=10)
    pb = RastriginInd(n_dims=10)
    ca, cb = arithmetic_crossover(pa, pb)
    for child in (ca, cb):
        assert np.all(child.genes >= -5.12) and np.all(child.genes <= 5.12)


def test_blx_crossover_bounds():
    pa = RastriginInd(n_dims=10)
    pb = RastriginInd(n_dims=10)
    ca, cb = blx_alpha_crossover(pa, pb)
    for child in (ca, cb):
        assert np.all(child.genes >= -5.12) and np.all(child.genes <= 5.12)


def test_gaussian_mutation_bounds():
    ind = RastriginInd(n_dims=10)
    for _ in range(50):
        gaussian_mutation(ind, sigma=0.5, mutation_rate=1.0)
    assert np.all(ind.genes >= -5.12) and np.all(ind.genes <= 5.12)


def test_gaussian_mutation_invalida_fitness():
    ind = RastriginInd(n_dims=5)
    _ = ind.fitness
    gaussian_mutation(ind, sigma=0.1, mutation_rate=1.0)
    assert ind._fitness is None


def test_create_rastrigin_population():
    pop = create_rastrigin_population(n_dims=5, size=30)
    assert len(pop) == 30
    assert all(isinstance(ind, RastriginInd) for ind in pop)


def test_engine_nqueens_sem_regressao():
    from problems.nqueens.individual import NQueensIndividual
    pop = [NQueensIndividual(4) for _ in range(30)]
    result = run(pop, max_generations=500, elite_size=2, mutation_rate=0.3)
    assert result.solved
    assert result.solution.fitness == 0
