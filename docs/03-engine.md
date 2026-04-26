# Fase 3 — Engine do Algoritmo Genético

O engine é o **orquestrador** — ele combina todos os operadores em um ciclo que repete até encontrar uma solução ou atingir o limite de gerações.

---

## Estrutura de dados do resultado

```python
@dataclass
class GAResult:
    solved: bool
    solution: Individual | None
    generation: int
    best_fitness_history: list[int]
```

`@dataclass` gera automaticamente `__init__`, `__repr__` e `__eq__` a partir dos campos declarados. É uma forma compacta de criar classes que apenas agrupam dados.

- `solved`: `True` se o AG encontrou uma solução com fitness 0
- `solution`: o indivíduo solução (ou `None` se não resolveu)
- `generation`: em qual geração convergiu (ou `max_generations` se não convergiu)
- `best_fitness_history`: lista com o melhor fitness de cada geração — útil para plotar a curva de evolução

---

## O loop principal — `run()`

```python
def run(
    population: list[Individual],
    max_generations: int = 1000,
    elite_size: int = 2,
    mutation_rate: float = 0.1,
    verbose: bool = False,
) -> GAResult:
```

Recebe a população inicial (já criada por `create_population`) e os hiperparâmetros. Retorna um `GAResult`.

### Passo 1 — Ordenar por fitness

```python
population.sort(key=lambda ind: ind.fitness)
best = population[0]
best_fitness_history.append(best.fitness)
```

`list.sort()` ordena in-place em ordem crescente de fitness (menor = melhor). Após a ordenação, `population[0]` é sempre o melhor indivíduo da geração atual.

O acesso a `ind.fitness` aqui dispara o cálculo para todos os indivíduos que ainda não foram avaliados (o cache garante que nenhum é calculado duas vezes na mesma geração).

### Passo 2 — Verificar convergência

```python
if best.fitness == 0:
    return GAResult(
        solved=True,
        solution=best.copy(),
        generation=generation,
        best_fitness_history=best_fitness_history,
    )
```

Se o melhor indivíduo tem fitness 0, encontramos uma solução sem conflitos. Retornamos imediatamente — não há razão para continuar. `best.copy()` garante que a solução retornada não é afetada por nenhuma modificação futura.

### Passo 3 — Elitismo

```python
elite = [ind.copy() for ind in population[:elite_size]]
```

Os `elite_size` melhores indivíduos da geração atual são copiados diretamente para a próxima **sem nenhuma modificação**. Isso garante que a melhor solução já encontrada nunca se perde.

**Sem elitismo**, um crossover ou mutação azarado poderia destruir o melhor indivíduo e o AG teria que "redescobrir" aquela solução. Com elitismo, o fitness mínimo da população é **monotonicamente não-crescente**.

### Passo 4 — Gerar filhos

```python
n_offspring = pop_size - elite_size
offspring: list[Individual] = []

while len(offspring) < n_offspring:
    parent_a, parent_b = roulette_select(population, 2)
    child_a, child_b = single_point_crossover(parent_a, parent_b)

    if random.random() < mutation_rate:
        swap_mutation(child_a)
    if random.random() < mutation_rate:
        swap_mutation(child_b)

    offspring.append(child_a)
    if len(offspring) < n_offspring:
        offspring.append(child_b)
```

Geramos exatamente `pop_size - elite_size` filhos para manter a população sempre do mesmo tamanho.

O crossover produz 2 filhos por chamada. Se `n_offspring` for ímpar, o segundo filho do último par é descartado (o `if len(offspring) < n_offspring` cuida disso).

`random.random()` retorna um float uniforme em `[0, 1)`. A mutação é aplicada de forma independente a cada filho — um pode ser mutado e o outro não.

### Passo 5 — Montar nova geração

```python
population = elite + offspring
```

A nova população tem exatamente `pop_size` indivíduos: os `elite_size` melhores preservados + os filhos gerados. O ciclo recomeça.

### Caso de não-convergência

```python
# Não convergiu dentro do limite de gerações
population.sort(key=lambda ind: ind.fitness)
return GAResult(
    solved=False,
    solution=None,
    generation=max_generations,
    best_fitness_history=best_fitness_history,
)
```

Se o loop terminar sem encontrar fitness 0, retornamos `solved=False` e `solution=None`. A ordenação final é feita apenas para deixar a população em estado consistente caso o chamador queira inspecioná-la.

---

## `app/ga/population.py` — Criação da população inicial

```python
def create_population(n_queens: int, size: int) -> list[Individual]:
    return [NQueensIndividual(n_queens) for _ in range(size)]
```

Gera `size` indivíduos com permutações aleatórias independentes. A diversidade inicial é importante: se todos os indivíduos começassem iguais, a evolução ficaria presa no mesmo ponto do espaço de busca.

---

## Diagrama do fluxo completo

```
create_population(n, size)
        │
        ▼
┌───────────────────────────────────┐
│  Para cada geração g = 1..max_g   │
│                                   │
│  1. Ordenar por fitness           │
│  2. Melhor fitness == 0?  ──Yes──►│──► GAResult(solved=True, g)
│  3. Copiar elite[:k]      ◄──No───│
│  4. Repetir até ter pop-k filhos: │
│     ├─ roulette_select(pop, 2)    │
│     ├─ single_point_crossover     │
│     └─ swap_mutation (prob. m)    │
│  5. population = elite + filhos   │
└───────────────────────────────────┘
        │ (max_g atingido)
        ▼
  GAResult(solved=False)
```

---

## Hiperparâmetros e seus efeitos

| Parâmetro | Muito baixo | Ideal | Muito alto |
|---|---|---|---|
| `pop_size` | Convergência prematura (pouca diversidade) | 30–100 | Lento, desperdício de memória |
| `elite_size` | Perde boas soluções | 2–5% do pop | O AG vira uma "cópia" sem evolução |
| `mutation_rate` | Estagna em ótimos locais | 0.1–0.3 | Busca aleatória, perde aprendizado |
| `max_generations` | Pode não convergir | 500–2000 | Apenas desperdício de tempo se já convergiu |

---

## Testes

```
tests/test_engine.py   7 testes
```

Casos importantes cobertos:
- Resolve 4 rainhas em até 500 gerações
- Resolve 8 rainhas em até 2000 gerações com mutação 20%
- O histórico de fitness tem comprimento igual ao número de gerações executadas
- O histórico de fitness é monotonicamente não-crescente (elitismo garantido)

```bash
.venv/Scripts/pytest tests/test_engine.py -v
```
