# Plano de Implementação: Otimização da Função Langermann

## Contexto

Extensão do framework GA para a **Função Langermann**, segunda função de benchmark contínuo. Segue exatamente a mesma arquitetura do Rastrigin: `fitness.py` + `individual.py` no pacote do problema, `create_*_population` em `ga/population.py`, CLI próprio em `app/`.

**Referência:** https://www.sfu.ca/~ssurjano/langer.html (Molga & Smutnicki, 2005)

**Função Langermann:**
```
f(x) = -Σᵢ₌₁ᵐ cᵢ · exp(-1/π · Σⱼ(xⱼ - Aᵢⱼ)²) · cos(π · Σⱼ(xⱼ - Aᵢⱼ)²)
```

**Parâmetros padrão (d=2):**
- m = 5
- c = [1, 2, 5, 2, 3]
- A = [[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]

**Domínio:** xᵢ ∈ [0, 10]  
**Mínimo global:** valor negativo (função altamente multimodal)  
**Característica:** muitos mínimos locais distribuídos irregularmente

---

## Arquivos a Criar

```
app/
├── problems/
│   └── langermann/
│       ├── __init__.py
│       ├── fitness.py          # langermann(genes, m, c, A) -> float
│       └── individual.py       # LangermannInd
└── langermann_main.py          # CLI entry point

tests/
└── test_langermann.py
```

## Arquivo a Modificar

| Arquivo | Mudança |
|---------|---------|
| `app/ga/population.py` | Adicionar `create_langermann_population` |

---

## Design: Parâmetros Variáveis de Dimensão

O caso padrão da literatura é d=2. Para d≠2, a matriz A precisa ter shape (m, d). Estratégia:

- Se `n_dims=2` → usa A e c padrão (benchmark clássico)
- Se `n_dims≠2` → gera A aleatório com shape `(m, n_dims)` e valores em [0, 10]

Todos os parâmetros (m, c, A) são armazenados no `LangermannInd` e repassados ao `fitness.py`.

---

## Passo a Passo Detalhado

### Passo 1 — `app/problems/langermann/__init__.py`
Arquivo vazio.

---

### Passo 2 — `app/problems/langermann/fitness.py`

```python
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
```

A negação no retorno torna o problema de minimização (mínimo global é o valor mais negativo).

---

### Passo 3 — `app/problems/langermann/individual.py`

```python
import numpy as np
from ga.individual import Individual
from problems.langermann.fitness import langermann, DEFAULT_M, DEFAULT_C, DEFAULT_A


class LangermannInd(Individual):
    def __init__(
        self,
        n_dims: int,
        genes: np.ndarray | None = None,
        bounds: np.ndarray | None = None,
        m: int = DEFAULT_M,
        c: np.ndarray = DEFAULT_C,
        A: np.ndarray | None = None,
    ):
        if bounds is None:
            bounds = np.full((n_dims, 2), [0.0, 10.0])
        if A is None:
            if n_dims == 2:
                A = DEFAULT_A.copy()
            else:
                A = np.random.uniform(0, 10, size=(m, n_dims))
        if genes is None:
            genes = np.random.uniform(bounds[:, 0], bounds[:, 1])
        super().__init__(genes)
        self.bounds = bounds
        self.m = m
        self.c = c
        self.A = A

    def _evaluate(self) -> float:
        return langermann(self.genes, self.m, self.c, self.A)

    def copy(self) -> "LangermannInd":
        return LangermannInd(
            len(self.genes),
            genes=self.genes.copy(),
            bounds=self.bounds.copy(),
            m=self.m,
            c=self.c.copy(),
            A=self.A.copy(),
        )
```

**Padrão idêntico ao `RastriginInd`**: herda de `Individual`, implementa `_evaluate()` e `copy()`. O `copy()` preserva todos os parâmetros (m, c, A) para que os operadores genéticos funcionem corretamente.

---

### Passo 4 — `app/ga/population.py`: adicionar helper Langermann

Adicionar ao final do arquivo (após `create_rastrigin_population`):

```python
from app.problems.langermann.individual import LangermannInd

def create_langermann_population(n_dims: int, size: int) -> list[Individual]:
    return [LangermannInd(n_dims) for _ in range(size)]
```

**Atenção:** o import deve usar `app.problems.langermann.individual` (com prefixo `app.`), seguindo o mesmo padrão do `RastriginInd`.

---

### Passo 5 — `app/langermann_main.py`: CLI

Segue o mesmo padrão de `rastrigin_main.py`:

```python
import argparse
import sys
import os

_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _app_dir)
sys.path.insert(0, os.path.dirname(_app_dir))

from ga.population import create_langermann_population
from ga.operators.crossover import blx_alpha_crossover
from ga.operators.mutation import gaussian_mutation
from ga.engine import run


def main():
    parser = argparse.ArgumentParser(description="Algoritmo Genético para Otimização da Função Langermann")
    parser.add_argument("-d", "--dims",    type=int,   default=2,    metavar="D",    help="Dimensões (padrão: 2)")
    parser.add_argument("-p", "--pop",     type=int,   default=200,  metavar="POP",  help="Tamanho da população (padrão: 200)")
    parser.add_argument("-g", "--gens",    type=int,   default=2000, metavar="GENS", help="Máximo de gerações (padrão: 2000)")
    parser.add_argument("-e", "--elite",   type=int,   default=2,    metavar="ELITE",help="Tamanho do elitismo (padrão: 2)")
    parser.add_argument("-s", "--sigma",   type=float, default=0.1,  metavar="S",    help="Desvio padrão da mutação gaussiana (padrão: 0.1)")
    parser.add_argument("-t", "--tol",     type=float, default=-4.0, metavar="TOL",  help="Tolerância (fitness < tol para convergência, padrão: -4.0)")
    parser.add_argument("-v", "--verbose", action="store_true",                      help="Exibe progresso por geração")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Algoritmo Genético — Langermann {args.dims}D")
    print(f"{'='*50}")
    print(f"  População: {args.pop} | Elite: {args.elite}")
    print(f"  Max gerações: {args.gens} | sigma: {args.sigma} | tol: {args.tol}")
    print(f"{'='*50}\n")

    population = create_langermann_population(args.dims, args.pop)

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

---

### Passo 6 — `tests/test_langermann.py`: testes

```python
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
    # Em regiões não triviais, a soma ponderada deve gerar valor negativo
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
```

---

## Sequência de Implementação

1. `app/problems/langermann/__init__.py`
2. `app/problems/langermann/fitness.py`
3. `app/problems/langermann/individual.py`
4. `app/ga/population.py` — adicionar `create_langermann_population`
5. `app/langermann_main.py`
6. `tests/test_langermann.py`

---

## Verificação

```bash
# Testes (51 no total: 42 existentes + 9 novos)
pytest tests/ -v

# Langermann 2D com parâmetros padrão
python app/langermann_main.py -d 2 -p 200 -g 2000 -v

# Parametrizado: n=20, elite=4, nGer=2000
python app/langermann_main.py -d 20 -e 4 -g 2000 -v
```
