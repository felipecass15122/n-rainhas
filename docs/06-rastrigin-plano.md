# Plano de Implementação: Otimização da Função Rastrigin

## Contexto

O framework GA atual suporta apenas N-Queens com codificação por permutação. O objetivo é estender o framework para **otimização contínua com codificação real**, implementando a **Função Rastrigin** como primeiro benchmark.

**Função Rastrigin:**
- Fórmula: `f(x) = A·n + Σᵢ[xᵢ² - A·cos(2π·xᵢ)]`, onde A = 10
- Domínio: xᵢ ∈ [-5.12, 5.12] para todo i
- Mínimo global: f(**0**) = 0.0
- Característica: altamente multimodal com mínimos locais regularmente distribuídos

---

## Novos Operadores Genéticos

### Crossover Aritmético (`arithmetic_crossover`)
Para dois pais p_a e p_b, com α ∈ [0, 1] sorteado aleatoriamente:
```
filho_1 = α · p_a + (1-α) · p_b
filho_2 = (1-α) · p_a + α · p_b
```
Garante que os filhos estejam dentro do casco convexo dos pais (sem necessidade de clamp).

### Crossover BLX-α (`blx_alpha_crossover`)
Para cada posição de gene i:
```
lo_i = min(p_a_i, p_b_i) - α · I_i
hi_i = max(p_a_i, p_b_i) + α · I_i
filho_i ~ U(lo_i, hi_i)
```
Onde `I_i = max - min`. α = 0.5 padrão. Genes são clampados nos bounds do indivíduo.

### Mutação Gaussiana (`gaussian_mutation`)
Para cada gene com probabilidade `mutation_rate`:
```
xᵢ' = xᵢ + N(0, σ)
xᵢ' = clip(xᵢ', lb_i, ub_i)
```
σ recomendado: `(ub - lb) × 0.1`. `mutation_rate` padrão: `1/n_dims`.

---

## Arquivos a Modificar

| Arquivo | Mudanças |
|---------|----------|
| `app/ga/individual.py` | Corrigir `_fitness: int \| None` → `float \| None`; `_evaluate()` e `fitness` → `float` |
| `app/ga/population.py` | Corrigir bug da função duplicada; adicionar `create_rastrigin_population` |
| `app/ga/engine.py` | Parâmetros opcionais `crossover_fn`, `mutation_fn`, `convergence_fn`; `list[int]` → `list[float]` |
| `app/ga/operators/crossover.py` | Adicionar `arithmetic_crossover` e `blx_alpha_crossover` |
| `app/ga/operators/mutation.py` | Adicionar `gaussian_mutation` |

---

## Arquivos a Criar

```
app/
├── problems/
│   └── rastrigin/
│       ├── __init__.py
│       ├── fitness.py          # rastrigin(genes) -> float
│       └── individual.py       # RastriginInd (sem factory)
└── rastrigin_main.py           # CLI entry point

tests/
└── test_rastrigin.py
```

---

## Design: Fluxo dos Bounds

`Individual` já tem `self.bounds: np.ndarray | None = None`. Os operadores verificam via duck-typing:

```python
if child.bounds is not None:
    child.genes = np.clip(child.genes, child.bounds[:, 0], child.bounds[:, 1])
```

`NQueensIndividual` herda `bounds = None` → operadores existentes não são afetados.

---

## Design: Criação de População

O padrão do projeto é ter funções de criação de população em `ga/population.py`:

```python
# padrão existente (N-Queens)
population = create_population(n_queens, pop_size)   # ga/population.py

# novo padrão (Rastrigin) — mesmo módulo, mesmo estilo
population = create_rastrigin_population(n_dims, pop_size)   # ga/population.py
```

**Não há `IndividualFactory` nem `RastriginIndFactory`** — a criação de população fica em `population.py`, como no padrão N-Queens.

---

## Design: Generalização do Engine

O engine atual tem três hardcodes a remover:

**Problema 1 — Operadores hardcoded (linhas 53-58):**
```python
child_a, child_b = single_point_crossover(parent_a, parent_b)
if random.random() < mutation_rate:
    swap_mutation(child_a)
if random.random() < mutation_rate:
    swap_mutation(child_b)
```

**Problema 2 — Convergência hardcoded (linha 36):**
```python
if best.fitness == 0:
```

**Problema 3 — Tipo do histórico (linha 14-15):**
```python
best_fitness_history: list[int]
```

**Solução:** três parâmetros opcionais com `None` como default:

```python
def run(
    population: list[Individual],
    max_generations: int = 1000,
    elite_size: int = 2,
    mutation_rate: float = 0.1,
    verbose: bool = False,
    crossover_fn=None,      # None → single_point_crossover
    mutation_fn=None,       # None → swap_mutation com prob=mutation_rate
    convergence_fn=None,    # None → lambda ind: ind.fitness == 0
) -> GAResult:
```

Defaults resolvidos no início do corpo de `run()`:
```python
if crossover_fn is None:
    crossover_fn = single_point_crossover
if mutation_fn is None:
    mutation_fn = lambda ind: swap_mutation(ind) if random.random() < mutation_rate else ind
if convergence_fn is None:
    convergence_fn = lambda ind: ind.fitness == 0
```

Loop interno:
```python
child_a, child_b = crossover_fn(parent_a, parent_b)
mutation_fn(child_a)
mutation_fn(child_b)
```

`gaussian_mutation` controla probabilidade por gene internamente — por isso `mutation_fn` é chamado incondicionalmente.

---

## Sequência de Implementação

1. `app/ga/individual.py` — corrigir tipos (`int` → `float`)
2. `app/ga/population.py` — corrigir bug + adicionar `create_rastrigin_population`
3. `app/ga/engine.py` — generalizar operadores e convergência
4. `app/ga/operators/crossover.py` — crossovers reais
5. `app/ga/operators/mutation.py` — mutação gaussiana
6. `app/problems/rastrigin/` — fitness e `RastriginInd`
7. `app/rastrigin_main.py` — CLI
8. `tests/test_rastrigin.py` — testes

---

## Passo a Passo Detalhado

### Passo 1 — `app/ga/individual.py`: corrigir tipos

**Estado atual:**
```python
self._fitness: int | None = None

@property
def fitness(self) -> int:

@abstractmethod
def _evaluate(self) -> int:
```

**O que fazer:** Trocar `int` por `float` nas três ocorrências acima. Nenhuma outra mudança.

**Verificação:** `pytest tests/ -v` — todos os 32 testes passam (`int` é subtipo de `float`).

---

### Passo 2 — `app/ga/population.py`: corrigir bug + adicionar helper Rastrigin

**Estado atual (bugado):**
```python
from problems.nqueens.individual import NQueensIndividual
from ga.individual import Individual

def create_population(n_queens: int, size: int) -> list[Individual]:
    return [NQueensIndividual(n_queens) for _ in range(size)]

def create_population(self, size: int) -> list["Individual"]:   # ← BUG: função duplicada com self
    return [self.create() for _ in range(size)]
```

**O que fazer:** Remover a segunda definição (inválida). Adicionar import de `RastriginInd` e a função `create_rastrigin_population`:

```python
from problems.nqueens.individual import NQueensIndividual
from problems.rastrigin.individual import RastriginInd
from ga.individual import Individual


def create_population(n_queens: int, size: int) -> list[Individual]:
    return [NQueensIndividual(n_queens) for _ in range(size)]


def create_rastrigin_population(n_dims: int, size: int) -> list[Individual]:
    return [RastriginInd(n_dims) for _ in range(size)]
```

**Verificação:** `pytest tests/ -v` — 32 testes passam.

---

### Passo 3 — `app/ga/engine.py`: generalizar operadores e convergência

**O que fazer:**

1. `GAResult.best_fitness_history: list[int]` → `list[float]`
2. `best_fitness_history: list[int] = []` → `list[float] = []`
3. Adicionar três parâmetros opcionais à assinatura de `run()`
4. Resolver defaults no início do corpo (após `pop_size = len(population)`)
5. Substituir `if best.fitness == 0:` → `if convergence_fn(best):`
6. Substituir bloco de crossover/mutação hardcoded pelas variáveis

**Verificação:** `pytest tests/ -v` — 32 testes passam (defaults mantêm comportamento N-Queens idêntico).

---

### Passo 4 — `app/ga/operators/crossover.py`: crossovers reais

Adicionar ao final do arquivo:

```python
def arithmetic_crossover(parent_a: Individual, parent_b: Individual) -> tuple[Individual, Individual]:
    alpha = np.random.rand()
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    child_a.genes = alpha * parent_a.genes + (1 - alpha) * parent_b.genes
    child_b.genes = (1 - alpha) * parent_a.genes + alpha * parent_b.genes
    for child in (child_a, child_b):
        if child.bounds is not None:
            child.genes = np.clip(child.genes, child.bounds[:, 0], child.bounds[:, 1])
        child.invalidate_fitness()
    return child_a, child_b


def blx_alpha_crossover(parent_a: Individual, parent_b: Individual, alpha: float = 0.5) -> tuple[Individual, Individual]:
    cmin = np.minimum(parent_a.genes, parent_b.genes)
    cmax = np.maximum(parent_a.genes, parent_b.genes)
    I = cmax - cmin
    lo = cmin - alpha * I
    hi = cmax + alpha * I
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    child_a.genes = np.random.uniform(lo, hi)
    child_b.genes = np.random.uniform(lo, hi)
    for child in (child_a, child_b):
        if child.bounds is not None:
            child.genes = np.clip(child.genes, child.bounds[:, 0], child.bounds[:, 1])
        child.invalidate_fitness()
    return child_a, child_b
```

`parent_a.copy()` preserva o tipo e os bounds — `RastriginInd.copy()` retorna `RastriginInd` com `bounds` copiados, então `child.bounds` já estará disponível.

---

### Passo 5 — `app/ga/operators/mutation.py`: mutação gaussiana

Adicionar ao final do arquivo:

```python
def gaussian_mutation(individual: Individual, sigma: float, mutation_rate: float | None = None) -> Individual:
    if mutation_rate is None:
        mutation_rate = 1.0 / len(individual.genes)
    mask = np.random.rand(len(individual.genes)) < mutation_rate
    individual.genes[mask] += np.random.normal(0, sigma, mask.sum())
    if individual.bounds is not None:
        individual.genes = np.clip(
            individual.genes,
            individual.bounds[:, 0],
            individual.bounds[:, 1],
        )
    individual.invalidate_fitness()
    return individual
```

Controla probabilidade por gene internamente via `mask` — diferente de `swap_mutation`. No engine, `mutation_fn(child)` é chamado incondicionalmente.

---

### Passo 6 — `app/problems/rastrigin/`: fitness e `RastriginInd`

#### `app/problems/rastrigin/__init__.py`
Arquivo vazio (marca o pacote).

#### `app/problems/rastrigin/fitness.py`
```python
import numpy as np

A = 10

def rastrigin(genes: np.ndarray) -> float:
    n = len(genes)
    return float(A * n + np.sum(genes**2 - A * np.cos(2 * np.pi * genes)))
```

#### `app/problems/rastrigin/individual.py`
```python
import numpy as np
from ga.individual import Individual
from problems.rastrigin.fitness import rastrigin


class RastriginInd(Individual):
    def __init__(self, n_dims: int, genes: np.ndarray | None = None, bounds: np.ndarray | None = None):
        if bounds is None:
            bounds = np.full((n_dims, 2), [-5.12, 5.12])
        if genes is None:
            genes = np.random.uniform(bounds[:, 0], bounds[:, 1])
        super().__init__(genes)
        self.bounds = bounds

    def _evaluate(self) -> float:
        return rastrigin(self.genes)

    def copy(self) -> "RastriginInd":
        return RastriginInd(len(self.genes), genes=self.genes.copy(), bounds=self.bounds.copy())
```

Segue o mesmo padrão de `NQueensIndividual`: herda de `Individual`, implementa `_evaluate()` e `copy()`. Sem factory.

---

### Passo 7 — `app/rastrigin_main.py`: CLI

```python
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ga.population import create_rastrigin_population
from ga.operators.crossover import blx_alpha_crossover
from ga.operators.mutation import gaussian_mutation
from ga.engine import run


def main():
    parser = argparse.ArgumentParser(description="Algoritmo Genético para Otimização da Função Rastrigin")
    parser.add_argument("-d", "--dims",    type=int,   default=10,   metavar="D",    help="Dimensões (padrão: 10)")
    parser.add_argument("-p", "--pop",     type=int,   default=200,  metavar="POP",  help="Tamanho da população (padrão: 200)")
    parser.add_argument("-g", "--gens",    type=int,   default=3000, metavar="GENS", help="Máximo de gerações (padrão: 3000)")
    parser.add_argument("-e", "--elite",   type=int,   default=2,    metavar="ELITE",help="Tamanho do elitismo (padrão: 2)")
    parser.add_argument("-s", "--sigma",   type=float, default=0.1,  metavar="S",    help="Desvio padrão da mutação gaussiana (padrão: 0.1)")
    parser.add_argument("-t", "--tol",     type=float, default=1e-6, metavar="TOL",  help="Tolerância para convergência (padrão: 1e-6)")
    parser.add_argument("-v", "--verbose", action="store_true",                      help="Exibe progresso por geração")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Algoritmo Genético — Rastrigin {args.dims}D")
    print(f"{'='*50}")
    print(f"  População: {args.pop} | Elite: {args.elite}")
    print(f"  Max gerações: {args.gens} | sigma: {args.sigma} | tol: {args.tol}")
    print(f"{'='*50}\n")

    population = create_rastrigin_population(args.dims, args.pop)

    result = run(
        population,
        max_generations=args.gens,
        elite_size=args.elite,
        mutation_rate=0.0,
        verbose=args.verbose,
        crossover_fn=blx_alpha_crossover,
        mutation_fn=lambda ind: gaussian_mutation(ind, sigma=args.sigma, mutation_rate=1 / args.dims),
        convergence_fn=lambda ind: ind.fitness < args.tol,
    )

    if result.solved:
        print(f"\nConvergiu na geração {result.generation}!")
    else:
        print(f"\nNao convergiu em {args.gens} geracoes.")

    print(f"Fitness final: {result.best_fitness_history[-1]:.8f}")
    if result.solution is not None:
        print(f"Genes:         {result.solution.genes}")


if __name__ == "__main__":
    main()
```

Segue o mesmo padrão de `main.py`: `sys.path.insert`, importa `create_*_population` de `ga.population`, chama `run()`.

---

### Passo 8 — `tests/test_rastrigin.py`: testes

```python
import numpy as np
import pytest
from problems.rastrigin.fitness import rastrigin
from problems.rastrigin.individual import RastriginInd
from ga.population import create_rastrigin_population
from ga.operators.crossover import arithmetic_crossover, blx_alpha_crossover
from ga.operators.mutation import gaussian_mutation
from ga.engine import run


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
    """Garante que a generalização do engine não quebrou o N-Queens."""
    from problems.nqueens.individual import NQueensIndividual
    pop = [NQueensIndividual(4) for _ in range(30)]
    result = run(pop, max_generations=500, elite_size=2, mutation_rate=0.3)
    assert result.solved
    assert result.solution.fitness == 0
```

---

## Verificação

```bash
# Regressão completa (32 existentes + 9 novos)
pytest tests/ -v

# Rastrigin 2D (pipeline rápido para validar)
python app/rastrigin_main.py -d 2 -p 100 -g 1000 -v

# Rastrigin 10D (benchmark padrão)
python app/rastrigin_main.py -d 10 -p 200 -g 3000 -s 0.1 -v
```

**Critério de sucesso:**
- Todos os 32 testes existentes passam sem alteração
- Os 9 novos testes de Rastrigin passam
- Engine converge para f < 1e-6 em 2D
- Fitness mínimo cai monotonicamente (garantido pelo elitismo)
