# Fase 2 — Operadores Genéticos

Os operadores genéticos são os mecanismos que fazem a população **evoluir** de geração em geração. São três: seleção, crossover e mutação.

---

## Seleção por Roleta Viciada

**Arquivo:** `app/ga/operators/selection.py`

A seleção decide quais indivíduos têm o "direito" de gerar filhos. A ideia da roleta é que todos participam, mas os melhores têm mais chances — como uma rifa onde quem tem melhor fitness comprou mais bilhetes.

```python
def roulette_select(population: list[Individual], n: int) -> list[Individual]:
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)
    weights = 1.0 / (fitnesses + 1.0)
    total = weights.sum()
    probabilities = weights / total

    chosen_indices = np.random.choice(len(population), size=n, replace=True, p=probabilities)
    return [population[i].copy() for i in chosen_indices]
```

### Por que `1 / (fitness + 1)` e não `1 / fitness`?

Nosso problema é de **minimização** (queremos fitness = 0). Para transformar "menor é melhor" em probabilidade (onde "maior é melhor"), invertemos:

```
fitness = 0 → peso = 1/(0+1) = 1.0   ← máxima chance
fitness = 1 → peso = 1/(1+1) = 0.5
fitness = 3 → peso = 1/(3+1) = 0.25
fitness = 9 → peso = 1/(9+1) = 0.1   ← mínima chance
```

O `+1` evita divisão por zero quando fitness = 0.

### Exemplo com população de 4 indivíduos

```
Indivíduo | Fitness | Peso (1/f+1) | Probabilidade
    A     |    0    |    1.000     |    44.4%
    B     |    1    |    0.500     |    22.2%
    C     |    2    |    0.333     |    14.8%
    D     |    3    |    0.250     |    11.1%
    E     |    4    |    0.200     |     8.9%
                      ─────────      ──────────
                       2.283          100%
```

`np.random.choice(..., p=probabilities)` sorteia índices com essas probabilidades. `replace=True` significa que o mesmo indivíduo pode ser escolhido mais de uma vez (seleção com reposição).

Por fim, retornamos **cópias** — nunca os indivíduos originais — para que o crossover possa modificá-los sem alterar a população atual.

---

## Crossover de Ponto Único com Reparação

**Arquivo:** `app/ga/operators/crossover.py`

O crossover imita a recombinação genética biológica: dois pais "trocam" partes do seu material genético para gerar filhos que herdam características de ambos.

```python
def single_point_crossover(parent_a: Individual, parent_b: Individual) -> tuple[Individual, Individual]:
    n = len(parent_a.genes)
    point = np.random.randint(1, n)

    child_a_genes = np.concatenate([parent_a.genes[:point], parent_b.genes[point:]])
    child_b_genes = np.concatenate([parent_b.genes[:point], parent_a.genes[point:]])

    child_a_genes = _repair(child_a_genes)
    child_b_genes = _repair(child_b_genes)

    child_a = parent_a.copy()
    child_a.genes = child_a_genes
    child_a.invalidate_fitness()

    child_b = parent_a.copy()
    child_b.genes = child_b_genes
    child_b.invalidate_fitness()

    return child_a, child_b
```

### O ponto de corte

`np.random.randint(1, n)` gera um número entre 1 e n-1 (inclusive). Evitamos 0 e n porque resultariam em filhos idênticos aos pais.

```
Pai A: [3, 6, 2, 7, 1, 4, 0, 5]
Pai B: [0, 4, 7, 5, 2, 6, 1, 3]
              ↑ point = 3

Filho A: [3, 6, 2 | 5, 2, 6, 1, 3]  ← primeiros 3 de A + últimos 5 de B
Filho B: [0, 4, 7 | 7, 1, 4, 0, 5]  ← primeiros 3 de B + últimos 5 de A
```

### O problema das duplicatas

Após o corte, os filhos geralmente têm genes repetidos e genes ausentes — não são mais permutações válidas:

```
Filho A: [3, 6, 2, 5, 2, 6, 1, 3]
                     ↑     ↑     ↑  duplicatas: 2, 6, 3
                     ausentes: 0, 4, 7
```

### A função `_repair`

```python
def _repair(genes: np.ndarray) -> np.ndarray:
    n = len(genes)
    seen = set()
    duplicates = []
    missing = []

    for i, gene in enumerate(genes):
        if gene in seen:
            duplicates.append(i)
        else:
            seen.add(gene)

    for val in range(n):
        if val not in seen:
            missing.append(val)

    result = genes.copy()
    for idx, val in zip(duplicates, missing):
        result[idx] = val

    return result
```

A lógica passo a passo:

1. **Primeira passagem:** percorre os genes da esquerda para direita. A primeira ocorrência de cada valor é mantida (adicionada a `seen`). As posições onde o valor já apareceu antes são marcadas em `duplicates`.

2. **Segunda passagem:** identifica os valores de `[0..n-1]` que não estão em `seen` — esses são os `missing`.

3. **Reparação:** substitui cada posição duplicada pelo próximo valor ausente.

```
Antes:     [3, 6, 2, 5, 2, 6, 1, 3]
seen após passagem:  {3, 6, 2, 5}
duplicates (índices): [4, 5, 7]   ← segunda ocorrência de 2, 6, 3
missing (valores):    [0, 4, 7]

Substituição:
  índice 4: 2 → 0
  índice 5: 6 → 4
  índice 7: 3 → 7

Depois:    [3, 6, 2, 5, 0, 4, 1, 7]  ✓ permutação válida
```

### Por que copiar do pai e não criar um novo indivíduo?

```python
child_a = parent_a.copy()
child_a.genes = child_a_genes
child_a.invalidate_fitness()
```

Copiamos o pai para herdar o tipo concreto (`NQueensIndividual`, ou qualquer outro problema no futuro). Em seguida, substituímos os genes e invalidamos o cache de fitness — o filho é um novo indivíduo diferente do pai.

---

## Mutação por Swap

**Arquivo:** `app/ga/operators/mutation.py`

A mutação introduz variação aleatória para que o AG não fique preso em ótimos locais.

```python
def swap_mutation(individual: Individual) -> Individual:
    n = len(individual.genes)
    i, j = np.random.choice(n, size=2, replace=False)
    individual.genes[i], individual.genes[j] = individual.genes[j], individual.genes[i]
    individual.invalidate_fitness()
    return individual
```

`np.random.choice(n, size=2, replace=False)` sorteia dois índices **diferentes** (sem reposição). Trocamos os valores nessas duas posições:

```
Antes:  [3, 6, 2, 5, 0, 4, 1, 7]
                ↑           ↑  i=2, j=6

Depois: [3, 6, 1, 5, 0, 4, 2, 7]
```

A troca mantém a permutação válida automaticamente — não inserimos nem removemos valores, só mudamos de posição.

`invalidate_fitness()` é chamado porque os genes mudaram — o fitness anterior não é mais válido.

A função retorna `individual` (o mesmo objeto, modificado in-place) para permitir encadeamento como `swap_mutation(child_a)` sem precisar de uma variável extra.

### Por que manter a taxa de mutação baixa?

Alta mutação transforma o AG em busca aleatória — perde a "memória" do que já aprendeu. Baixa demais e o AG converge para ótimos locais sem explorar outras regiões. O valor padrão de 20% (`--mutation 0.2`) é um bom equilíbrio para o problema das rainhas.

---

## Testes

```
tests/test_operators.py   13 testes
```

Casos importantes cobertos:
- Roleta seleciona o número correto de indivíduos e retorna cópias
- Indivíduo com fitness 0 é selecionado mais frequentemente que os ruins
- `_repair` lida com zero, uma ou muitas duplicatas
- Crossover produz permutações válidas em 20 execuções consecutivas
- Pais não são modificados pelo crossover
- Swap mantém a permutação válida e invalida o cache de fitness

```bash
.venv/Scripts/pytest tests/test_operators.py -v
```
