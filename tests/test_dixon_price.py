import numpy as np
import pytest
from app.problems.dixon_price.fitness import dixon_price
from app.problems.dixon_price.individual import DixonPriceInd
from app.ga.population import create_dixon_price_population
from app.ga.operators.crossover import arithmetic_crossover, blx_alpha_crossover
from app.ga.operators.mutation import gaussian_mutation


def test_dixon_price_output_is_float():
    genes = np.array([1.0, 0.5])
    result = dixon_price(genes)
    assert isinstance(result, float)


def test_dixon_price_no_otimo_global():
    # x* = [1.0, 1/sqrt(2)] para d=2 → f = 0
    x_opt = np.array([1.0, 1.0 / np.sqrt(2)])
    assert dixon_price(x_opt) == pytest.approx(0.0, abs=1e-10)


def test_dixon_price_ind_dentro_dos_bounds():
    ind = DixonPriceInd(n_dims=5)
    assert np.all(ind.genes >= -10.0) and np.all(ind.genes <= 10.0)


def test_dixon_price_ind_copy_e_independente():
    ind = DixonPriceInd(n_dims=3)
    clone = ind.copy()
    clone.genes[0] = 999.0
    assert ind.genes[0] != 999.0


def test_dixon_price_ind_copy_preserva_bounds():
    ind = DixonPriceInd(n_dims=3)
    clone = ind.copy()
    assert np.array_equal(clone.bounds, ind.bounds)


def test_arithmetic_crossover_bounds_dixon_price():
    pa = DixonPriceInd(n_dims=3)
    pb = DixonPriceInd(n_dims=3)
    ca, cb = arithmetic_crossover(pa, pb)
    for child in (ca, cb):
        assert np.all(child.genes >= -10.0) and np.all(child.genes <= 10.0)


def test_blx_crossover_bounds_dixon_price():
    pa = DixonPriceInd(n_dims=3)
    pb = DixonPriceInd(n_dims=3)
    ca, cb = blx_alpha_crossover(pa, pb)
    for child in (ca, cb):
        assert np.all(child.genes >= -10.0) and np.all(child.genes <= 10.0)


def test_gaussian_mutation_bounds_dixon_price():
    ind = DixonPriceInd(n_dims=3)
    for _ in range(50):
        gaussian_mutation(ind, sigma=2.0, mutation_rate=1.0)
    assert np.all(ind.genes >= -10.0) and np.all(ind.genes <= 10.0)


def test_create_dixon_price_population():
    pop = create_dixon_price_population(n_dims=3, size=30)
    assert len(pop) == 30
    assert all(isinstance(ind, DixonPriceInd) for ind in pop)
