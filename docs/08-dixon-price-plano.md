# Plano de Implementação: Otimização da Função Dixon-Price

## Contexto

Extensão do framework GA para a **Função Dixon-Price**, terceira função de benchmark contínuo. Segue exatamente a mesma arquitetura do Rastrigin e Langermann: `fitness.py` + `individual.py` no pacote do problema, `create_*_population` em `ga/population.py`, CLI próprio em `app/`.

**Referência:** https://www.sfu.ca/~ssurjano/dixonpr.html

**Função Dixon-Price:**
```
f(x) = (x₁ - 1)² + Σᵢ₌₂ⁿ i · (2·xᵢ² - xᵢ₋₁)²
```

**Domínio:** xᵢ ∈ [-10, 10]  
**Mínimo global:** f(x*) = 0  
**Ponto ótimo:** x*ᵢ = 2^(-(2^i - 2) / 2^i)

Exemplos para d=2:
- x*₁ = 1.0
- x*₂ = ±1/√2 ≈ ±0.7071 (simetria quadrática em x₂)

**Característica:** convexa com um único vale profundo — difícil de refinar para precisão alta por causa do gradiente muito suave próximo ao ótimo.

---

## Arquivos Criados

```
app/
├── problems/
│   └── dixon_price/
│       ├── __init__.py
│       ├── fitness.py          # dixon_price(genes) -> float
│       └── individual.py       # DixonPriceInd
└── dixon_price_main.py         # CLI entry point

tests/
└── test_dixon_price.py
```

## Arquivo Modificado

| Arquivo | Mudança |
|---------|---------|
| `app/ga/population.py` | Adicionado `create_dixon_price_population` |

---

## Implementação

### `app/problems/dixon_price/fitness.py`

```python
import numpy as np

def dixon_price(genes: np.ndarray) -> float:
    n = len(genes)
    term1 = (genes[0] - 1.0) ** 2
    i = np.arange(2, n + 1, dtype=float)
    term2 = np.sum(i * (2.0 * genes[1:] ** 2 - genes[:-1]) ** 2)
    return float(term1 + term2)
```

Implementação vetorizada: `np.arange(2, n+1)` gera os pesos `i` de uma vez, evitando loop Python.

---

### `app/problems/dixon_price/individual.py`

```python
import numpy as np
from ga.individual import Individual
from problems.dixon_price.fitness import dixon_price

class DixonPriceInd(Individual):
    def __init__(self, n_dims: int, genes: np.ndarray | None = None, bounds: np.ndarray | None = None):
        if bounds is None:
            bounds = np.full((n_dims, 2), [-10.0, 10.0])
        if genes is None:
            genes = np.random.uniform(bounds[:, 0], bounds[:, 1])
        super().__init__(genes)
        self.bounds = bounds

    def _evaluate(self) -> float:
        return dixon_price(self.genes)

    def copy(self) -> "DixonPriceInd":
        return DixonPriceInd(len(self.genes), genes=self.genes.copy(), bounds=self.bounds.copy())
```

Mais simples que `LangermannInd` — sem parâmetros extras além de `bounds`, pois a função Dixon-Price não tem hiperparâmetros (m, c, A).

---

### `app/ga/population.py`: helper adicionado

```python
from app.problems.dixon_price.individual import DixonPriceInd

def create_dixon_price_population(n_dims: int, size: int) -> list[Individual]:
    return [DixonPriceInd(n_dims) for _ in range(size)]
```

---

### `app/dixon_price_main.py`: CLI

```python
import argparse
import sys
import os

_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _app_dir)
sys.path.insert(0, os.path.dirname(_app_dir))

from ga.population import create_dixon_price_population
from ga.operators.crossover import blx_alpha_crossover
from ga.operators.mutation import gaussian_mutation
from ga.engine import run


def main():
    parser = argparse.ArgumentParser(description="Algoritmo Genético para Otimização da Função Dixon-Price")
    parser.add_argument("-d", "--dims",    type=int,   default=20,   metavar="D",    help="Dimensões (padrão: 20)")
    parser.add_argument("-p", "--pop",     type=int,   default=200,  metavar="POP",  help="Tamanho da população (padrão: 200)")
    parser.add_argument("-g", "--gens",    type=int,   default=2000, metavar="GENS", help="Máximo de gerações (padrão: 2000)")
    parser.add_argument("-e", "--elite",   type=int,   default=4,    metavar="ELITE",help="Tamanho do elitismo (padrão: 4)")
    parser.add_argument("-s", "--sigma",   type=float, default=0.05, metavar="S",    help="Desvio padrão da mutação gaussiana (padrão: 0.05)")
    parser.add_argument("-t", "--tol",     type=float, default=1e-6, metavar="TOL",  help="Tolerância (fitness < tol para convergência, padrão: 1e-6)")
    parser.add_argument("-v", "--verbose", action="store_true",                      help="Exibe progresso por geração")
    args = parser.parse_args()
    ...

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
```

**Nota sobre sigma:** Dixon-Price exige `sigma=0.05` (vs. 0.1 do Rastrigin). Sigma maior impede o refinamento fino necessário para cruzar a barreira de 1e-6, pois a função tem gradiente muito suave próximo ao ótimo.

---

## Parâmetros Padrão de Execução

Todos os 3 problemas contínuos usam os mesmos defaults principais:

| Parâmetro | Valor |
|-----------|-------|
| `--dims`  | 20    |
| `--elite` | 4     |
| `--gens`  | 2000  |
| `--pop`   | 200   |

---

## Verificação

```bash
# Testes (60 no total: 51 existentes + 9 novos)
pytest tests/test_dixon_price.py -v
pytest tests/ -v

# Dixon-Price 2D (pipeline rápido para validar)
python app/dixon_price_main.py -d 2 -p 300 -e 5 -g 5000 -s 0.05 -v

# Dixon-Price 20D (benchmark padrão)
python app/dixon_price_main.py -v
```

**Critério de sucesso:**
- f(x*) = 0 em `np.array([1.0, 1/√2])` passa com `approx(0.0, abs=1e-10)`
- Todos os 60 testes passam
- CLI converge para f < 1e-6 com parâmetros ajustados
- x* ≈ [1.0, ±0.707] para d=2
