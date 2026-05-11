import numpy as np
import pytest
from app.problems.langermann.fitness import langermann, DEFAULT_M, DEFAULT_C, DEFAULT_A
from app.problems.langermann.individual import LangermannInd
from app.ga.population import create_langermann_population
from app.ga.operators.crossover import arithmetic_crossover, blx_alpha_crossover
from app.ga.operators.mutation import gaussian_mutation


def test_langermann_output_is_float():
    genes = np.array([2.0, 3.0])
    result = langermann(genes, DEFAULT_M, DEFAULT_C, DEFAULT_A)
    assert isinstance(result, float)


def test_langermann_is_negative():
    genes = np.array([3.0, 5.0])  # próximo de A[0]
    result = langermann(genes, DEFAULT_M, DEFAULT_C, DEFAULT_A)
    assert result < 0


def test_langermann_ind_dentro_dos_bounds():
    ind = LangermannInd(n_dims=2)
    assert np.all(ind.genes >= 0.0) and np.all(ind.genes <= 10.0)


def test_langermann_ind_copy_e_independente():
    ind = LangermannInd(n_dims=2)
    clone = ind.copy()
    clone.genes[0] = 999.0
    assert ind.genes[0] != 999.0


def test_langermann_ind_copy_preserva_parametros():
    ind = LangermannInd(n_dims=2)
    clone = ind.copy()
    assert clone.m == ind.m
    assert np.array_equal(clone.c, ind.c)
    assert np.array_equal(clone.A, ind.A)


def test_arithmetic_crossover_bounds_langermann():
    pa = LangermannInd(n_dims=2)
    pb = LangermannInd(n_dims=2)
    ca, cb = arithmetic_crossover(pa, pb)
    for child in (ca, cb):
        assert np.all(child.genes >= 0.0) and np.all(child.genes <= 10.0)


def test_blx_crossover_bounds_langermann():
    pa = LangermannInd(n_dims=2)
    pb = LangermannInd(n_dims=2)
    ca, cb = blx_alpha_crossover(pa, pb)
    for child in (ca, cb):
        assert np.all(child.genes >= 0.0) and np.all(child.genes <= 10.0)


def test_gaussian_mutation_bounds_langermann():
    ind = LangermannInd(n_dims=2)
    for _ in range(50):
        gaussian_mutation(ind, sigma=1.0, mutation_rate=1.0)
    assert np.all(ind.genes >= 0.0) and np.all(ind.genes <= 10.0)


def test_create_langermann_population():
    pop = create_langermann_population(n_dims=2, size=30)
    assert len(pop) == 30
    assert all(isinstance(ind, LangermannInd) for ind in pop)
