# Plano de Implementação: Otimização da Função Rastrigin

## Contexto

O framework GA atual suporta apenas o N-Queens com codificação por permutação. Este plano estende o framework para **otimização de funções contínuas com codificação real**, implementando a **Função Rastrigin** como primeiro benchmark.

**Função Rastrigin:**
- Fórmula: `f(x) = A·n + Σᵢ[xᵢ² - A·cos(2π·xᵢ)]`, onde A = 10
- Domínio: xᵢ ∈ [-5.12, 5.12] para todo i
- Mínimo global: f(**0**) = 0.0
- Característica: altamente multimodal com mínimos locais regularmente distribuídos

---

## Novos Operadores Genéticos

### Crossover Aritmético
Para dois pais p_a e p_b, com α ∈ [0, 1] sorteado aleatoriamente:
```
filho_1 = α · p_a + (1-α) · p_b
filho_2 = (1-α) · p_a + α · p_b
```
Garante que os filhos estejam dentro do casco convexo dos pais (sem necessidade de clamp).

### Crossover BLX-α (Blend Crossover)
Para cada posição de gene i:
```
cmin_i = min(p_a_i, p_b_i)
cmax_i = max(p_a_i, p_b_i)
I_i    = cmax_i - cmin_i
filho_i ~ U(cmin_i - α·I_i, cmax_i + α·I_i)
```
Amostra uma região estendida além dos pais. α = 0.5 é o valor padrão. Genes são clampados nos bounds do indivíduo.

### Mutação Gaussiana
Para cada gene com probabilidade `mutation_rate`:
```
xᵢ' = xᵢ + N(0, σ)
xᵢ' = clip(xᵢ', lb_i, ub_i)
```
σ recomendado: `(ub - lb) × 0.1` ou `1/n_dims` como probabilidade por gene.

---

## Arquivos a Modificar

| Arquivo | Mudanças |
|---------|----------|
| `app/ga/individual.py` | Widening de tipo `int → float`; adicionar `self.bounds = None`; adicionar classe `IndividualFactory` |
| `app/ga/engine.py` | Parâmetros opcionais `crossover_fn`, `mutation_fn`, `convergence_fn`; substitui hardcodes |
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
│       └── individual.py       # RastriginInd + RastriginIndFactory
└── rastrigin_main.py           # CLI entry point

tests/
└── test_rastrigin.py
```

---

## Design: Fluxo dos Bounds

Os bounds são armazenados em cada `Individual` como `self.bounds: np.ndarray | None` com shape `(n_dims, 2)`. Operadores acessam os bounds diretamente via duck-typing:

```python
if individual.bounds is not None:
    individual.genes = np.clip(individual.genes, bounds[:, 0], bounds[:, 1])
```

`NQueensIndividual` herda `bounds = None` da base e os operadores existentes não são afetados.

---

## Design: Generalização do Engine

Novos parâmetros opcionais com `None` como default preservam total retrocompatibilidade:

```python
def run(
    population,
    max_generations=1000,
    elite_size=2,
    mutation_rate=0.1,
    verbose=False,
    crossover_fn=None,      # None → single_point_crossover
    mutation_fn=None,       # None → swap_mutation
    convergence_fn=None,    # None → fitness == 0
) -> GAResult:
```

Para Rastrigin, os operadores são passados via closures que capturam `sigma` e `tolerance`:

```python
result = run(
    population,
    crossover_fn=blx_alpha_crossover,
    mutation_fn=lambda ind: gaussian_mutation(ind, sigma=0.1, mutation_rate=1/dims),
    convergence_fn=lambda ind: ind.fitness < 1e-6,
)
```

---

## Classes Principais

### `IndividualFactory` (nova classe abstrata em `app/ga/individual.py`)
```python
class IndividualFactory(ABC):
    @abstractmethod
    def create(self) -> Individual: ...

    def create_population(self, size: int) -> list[Individual]:
        return [self.create() for _ in range(size)]
```

### `RastriginInd`
```python
class RastriginInd(Individual):
    def __init__(self, n_dims, genes=None, bounds=None):
        # bounds: np.ndarray shape (n_dims, 2), default [[-5.12, 5.12]] * n_dims
        ...
    def _evaluate(self) -> float: ...
    def copy(self) -> "RastriginInd": ...
```

### `RastriginIndFactory`
```python
class RastriginIndFactory(IndividualFactory):
    def __init__(self, n_dims=10, bounds=(-5.12, 5.12)):
        self.n_dims = n_dims
        self.bounds = np.full((n_dims, 2), bounds)

    def create(self) -> RastriginInd:
        return RastriginInd(self.n_dims, bounds=self.bounds.copy())
```

---

## Sequência de Implementação

1. `app/ga/individual.py` — base muda primeiro
2. `app/ga/engine.py` — generalizar operadores e convergência
3. `app/ga/operators/crossover.py` — crossovers reais
4. `app/ga/operators/mutation.py` — mutação gaussiana
5. `app/problems/rastrigin/` — fitness, Individual, Factory
6. `app/rastrigin_main.py` — CLI
7. `tests/test_rastrigin.py` — testes

---

## Verificação

```bash
# Regressão: N-Queens continua funcionando
pytest tests/ -v

# Rastrigin 2D (convergência rápida, boa para validar)
python app/rastrigin_main.py -d 2 -p 100 -g 1000 -v

# Rastrigin 10D (caso padrão)
python app/rastrigin_main.py -d 10 -p 200 -g 3000 -s 0.1 -v
```

**Critério de sucesso:** testes existentes passam sem alteração; fitness cai monotonicamente; engine converge para f < 1e-6 em 2D/5D.
