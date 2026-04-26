# Fase 1 — Representação e Função de Aptidão

## O Problema das N-Rainhas

Posicionar N rainhas em um tabuleiro N×N de forma que nenhuma rainha ataque outra. Uma rainha ataca em todas as direções: linha, coluna e as duas diagonais.

Para N=8, existem 4.426.165.368 formas de posicionar 8 peças em 64 casas, mas apenas **92 soluções válidas**.

---

## Como representar uma solução como vetor de genes

A escolha da representação é a decisão mais importante do AG — ela define o espaço de busca inteiro.

**Representação ingênua:** um vetor binário de N² bits (1 = tem rainha, 0 = não tem). Problema: a maioria dos cromossomos seria inválida (mais ou menos de N rainhas, várias na mesma linha etc). O AG gastaria a maior parte do esforço descartando soluções estruturalmente inválidas.

**Nossa representação: permutação de [0..N-1]**

```
genes = [3, 6, 2, 7, 1, 4, 0, 5]
         ↑                    ↑
    linha 0               linha 7
    col 3                 col 5
```

`genes[i]` = coluna onde a rainha da **linha i** está posicionada.

Como cada índice aparece exatamente uma vez (é uma permutação), garantimos por construção:
- **Sem conflitos de linha** — há exatamente uma rainha por linha (cada índice `i` aparece uma vez)
- **Sem conflitos de coluna** — cada valor aparece uma vez, logo cada coluna tem no máximo uma rainha

Só precisamos verificar as **diagonais**. Isso reduz o espaço de busca de N^N para N!:

| N | N^N (ingênuo) | N! (permutação) |
|---|---|---|
| 8 | 16.777.216 | 40.320 |
| 12 | 8.916.100.448.256 | 479.001.600 |

---

## `app/ga/individual.py` — A classe base `Individual`

```python
from abc import ABC, abstractmethod
import numpy as np

class Individual(ABC):
    def __init__(self, genes: np.ndarray):
        self.genes = genes
        self._fitness: int | None = None
```

`ABC` (Abstract Base Class) força que toda subclasse implemente os métodos marcados com `@abstractmethod`. Isso garante que qualquer problema novo (ex: caixeiro-viajante, mochila) implemente a interface correta.

`self._fitness` começa como `None` — o fitness **não é calculado na criação**, só quando for acessado pela primeira vez.

---

### O padrão de cache (lazy evaluation)

```python
@property
def fitness(self) -> int:
    if self._fitness is None:
        self._fitness = self._evaluate()
    return self._fitness
```

`@property` faz `ind.fitness` funcionar como atributo (sem parênteses), mas executa código por baixo.

A lógica é: "se ainda não calculei, calculo agora e guardo. Da próxima vez, retorno o valor guardado." Isso evita recalcular o fitness do mesmo indivíduo várias vezes por geração.

```python
def invalidate_fitness(self):
    self._fitness = None
```

Quando os genes são modificados (mutação, crossover), o cache fica desatualizado. `invalidate_fitness()` sinaliza que o próximo acesso a `.fitness` deve recalcular do zero.

---

### Métodos abstratos

```python
@abstractmethod
def _evaluate(self) -> int:
    ...

@abstractmethod
def copy(self) -> "Individual":
    ...
```

Todo problema concreto deve implementar:
- `_evaluate()`: a lógica de aptidão específica do problema
- `copy()`: cópia profunda — necessária porque o AG modifica os filhos sem querer alterar os pais

---

## `app/problems/nqueens/individual.py` — O indivíduo concreto

```python
class NQueensIndividual(Individual):
    def __init__(self, n: int, genes: np.ndarray | None = None):
        self.n = n
        if genes is None:
            genes = np.random.permutation(n)
        super().__init__(genes)
```

`np.random.permutation(n)` gera um array com os valores `[0, 1, ..., n-1]` em ordem aleatória. Por exemplo, para n=4: `[2, 0, 3, 1]`.

O parâmetro `genes` é opcional — se não for passado, um indivíduo aleatório é gerado. Se for passado (ex: ao criar um filho no crossover), aquele vetor é usado diretamente.

```python
def _evaluate(self) -> int:
    return count_conflicts(self.genes)
```

Delega o cálculo para a função pura `count_conflicts`. Separar a lógica de fitness da classe do indivíduo facilita testar e reutilizar.

```python
def copy(self) -> "NQueensIndividual":
    clone = NQueensIndividual(self.n, self.genes.copy())
    return clone
```

`self.genes.copy()` cria um novo array NumPy com os mesmos valores. Sem o `.copy()`, `clone.genes` e `self.genes` apontariam para o mesmo array na memória — modificar um modificaria o outro.

---

## `app/problems/nqueens/fitness.py` — Contagem de conflitos

```python
def count_conflicts(genes: np.ndarray) -> int:
    n = len(genes)
    conflicts = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(int(genes[i]) - int(genes[j])) == abs(i - j):
                conflicts += 1
    return conflicts
```

Itera sobre todos os **pares não-repetidos** de rainhas `(i, j)` com `i < j` — isso é uma combinação C(n, 2).

A condição de conflito diagonal é:

```
|coluna_i - coluna_j| == |linha_i - linha_j|
```

Geometricamente: duas casas estão na mesma diagonal se a diferença horizontal entre elas é igual à diferença vertical.

### Exemplo passo a passo para genes = [1, 3, 0, 2] (solução de 4 rainhas)

```
Par (0,1): |genes[0]-genes[1]| = |1-3| = 2  |0-1| = 1  → 2 ≠ 1 → sem conflito
Par (0,2): |genes[0]-genes[2]| = |1-0| = 1  |0-2| = 2  → 1 ≠ 2 → sem conflito
Par (0,3): |genes[0]-genes[3]| = |1-2| = 1  |0-3| = 3  → 1 ≠ 3 → sem conflito
Par (1,2): |genes[1]-genes[2]| = |3-0| = 3  |1-2| = 1  → 3 ≠ 1 → sem conflito
Par (1,3): |genes[1]-genes[3]| = |3-2| = 1  |1-3| = 2  → 1 ≠ 2 → sem conflito
Par (2,3): |genes[2]-genes[3]| = |0-2| = 2  |2-3| = 1  → 2 ≠ 1 → sem conflito

Total de conflitos: 0 ✓
```

O `int()` em `abs(int(genes[i]) - int(genes[j]))` é necessário porque NumPy pode retornar tipos como `np.int64` que se comportam diferente do `int` do Python em certas operações de comparação.

---

## Testes

```
tests/test_nqueens_individual.py   13 testes
```

Casos importantes cobertos:
- Solução `[0,4,7,5,2,6,1,3]` tem fitness 0
- Diagonal principal `[0,1,2,3,4,5,6,7]` tem fitness 28 (todos os C(8,2) pares conflitam)
- `copy()` é realmente independente (modificar o clone não afeta o original)
- `invalidate_fitness()` força recálculo após modificação dos genes

```bash
.venv/Scripts/pytest tests/test_nqueens_individual.py -v
```
