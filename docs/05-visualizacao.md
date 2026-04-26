# Fase 5 — Visualização do Tabuleiro

**Arquivo:** `app/utils/visualizer.py`

A visualização transforma o vetor de genes em uma representação gráfica do tabuleiro no terminal.

---

## O código

```python
def print_board(genes: np.ndarray) -> None:
    n = len(genes)
    separator = "+" + ("---+" * n)

    print(separator)
    for row in range(n):
        queen_col = genes[row]
        cells = []
        for col in range(n):
            cells.append(" Q " if col == queen_col else " . ")
        print("|" + "|".join(cells) + "|")
        print(separator)
```

### Como funciona linha por linha

**`separator = "+" + ("---+" * n)`**

Para n=4 gera: `+---+---+---+---+`

`"---+"` repetido n vezes cria as divisões de colunas. O `+` no início fecha a borda esquerda.

**`for row in range(n)`**

Itera sobre cada linha do tabuleiro (de 0 a n-1).

**`queen_col = genes[row]`**

Lê a coluna da rainha nesta linha diretamente do vetor de genes. Por exemplo, se `genes[2] = 5`, a rainha da linha 2 está na coluna 5.

**`cells = [" Q " if col == queen_col else " . " for col in range(n)]`**

Para cada coluna da linha atual, decide se coloca `Q` (rainha) ou `.` (vazio). O resultado é uma lista de strings de 3 caracteres cada.

**`print("|" + "|".join(cells) + "|")`**

`"|".join(cells)` une as células com `|` entre elas. As bordas esquerda e direita são adicionadas manualmente:

```
cells = [" . ", " Q ", " . ", " . "]
"|".join(cells) = " . | Q | . | . "
resultado final  = "| . | Q | . | . |"
```

---

## Exemplo completo para genes = [1, 3, 0, 2]

```
genes[0] = 1  → rainha na linha 0, coluna 1
genes[1] = 3  → rainha na linha 1, coluna 3
genes[2] = 0  → rainha na linha 2, coluna 0
genes[3] = 2  → rainha na linha 3, coluna 2
```

Saída:

```
+---+---+---+---+
| . | Q | . | . |   ← linha 0: rainha na coluna 1
+---+---+---+---+
| . | . | . | Q |   ← linha 1: rainha na coluna 3
+---+---+---+---+
| Q | . | . | . |   ← linha 2: rainha na coluna 0
+---+---+---+---+
| . | . | Q | . |   ← linha 3: rainha na coluna 2
+---+---+---+---+
```

Nenhuma rainha está na mesma coluna, linha ou diagonal → solução válida (fitness = 0).

---

## Como verificar manualmente que é uma solução

Para confirmar que não há conflitos diagonais, trace as diagonais das rainhas:

```
Linha 0, Coluna 1:
  diagonal ↘: (0,1),(1,2),(2,3)  → não há rainha em (1,2) nem (2,3) ✓
  diagonal ↙: (0,1),(1,0)        → não há rainha em (1,0) ✓

Linha 1, Coluna 3:
  diagonal ↘: (1,3),(2,4)→fora  ✓
  diagonal ↙: (1,3),(2,2),(3,1) → não há rainha em (2,2) nem (3,1) ✓

Linha 2, Coluna 0:
  diagonal ↘: (2,0),(3,1)       → não há rainha em (3,1) ✓
  diagonal ↙: fora do tabuleiro ✓

Linha 3, Coluna 2:
  sem rainhas abaixo            ✓
```

Todas as verificações passam → solução válida.
