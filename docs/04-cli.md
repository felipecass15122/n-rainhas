# Fase 4 — Interface de Linha de Comando (CLI)

**Arquivo:** `app/main.py`

O `main.py` é o ponto de entrada do programa. Ele não contém lógica do AG — apenas lê os parâmetros do usuário, chama o engine e exibe o resultado.

---

## Estrutura geral

```python
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ga.population import create_population
from ga.engine import run
from utils.visualizer import print_board
```

`sys.path.insert(0, os.path.dirname(__file__))` adiciona o diretório `app/` ao início do caminho de busca de módulos Python. Isso permite que os imports funcionem tanto ao rodar `python app/main.py` da raiz do projeto quanto ao rodar de dentro do diretório `app/`.

---

## Definição dos argumentos com `argparse`

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algoritmo Genético para o Problema das N-Rainhas"
    )
    parser.add_argument("-n", "--queens",    type=int,   default=8,    ...)
    parser.add_argument("-p", "--pop",       type=int,   default=50,   ...)
    parser.add_argument("-e", "--elite",     type=int,   default=2,    ...)
    parser.add_argument("-g", "--generations",type=int,  default=1000, ...)
    parser.add_argument("-m", "--mutation",  type=float, default=0.2,  ...)
    parser.add_argument("-v", "--verbose",   action="store_true",      ...)
    return parser.parse_args()
```

`argparse` gera automaticamente a mensagem de ajuda (`--help`), valida os tipos (ex: `type=int` rejeita letras), e atribui os valores ao objeto `Namespace` retornado.

`action="store_true"` para `--verbose` significa: se a flag estiver presente, o valor é `True`; se ausente, `False`. Não precisa de valor (`--verbose True` seria errado — apenas `--verbose`).

Cada argumento tem duas formas:
- Forma curta (`-n`): para uso rápido no terminal
- Forma longa (`--queens`): auto-documentada e mais legível em scripts

---

## A função `main()`

```python
def main() -> None:
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  Algoritmo Genético — {args.queens}-Rainhas")
    ...

    population = create_population(args.queens, args.pop)
    result = run(
        population,
        max_generations=args.generations,
        elite_size=args.elite,
        mutation_rate=args.mutation,
        verbose=args.verbose,
    )

    if result.solved:
        print(f"\nSolução encontrada na geração {result.generation}!")
        print(f"Genes: {result.solution.genes.tolist()}")
        print()
        print_board(result.solution.genes)
    else:
        print(f"\nNão foi encontrada solução em {args.generations} gerações.")
        print("Tente aumentar o número de gerações (-g) ou a população (-p).")
```

O fluxo é linear:
1. Lê os argumentos
2. Exibe o cabeçalho com os parâmetros escolhidos
3. Cria a população inicial
4. Roda o AG
5. Exibe o resultado (solução ou mensagem de falha)

```python
if __name__ == "__main__":
    main()
```

`__name__ == "__main__"` é `True` apenas quando o arquivo é executado diretamente (`python app/main.py`). Se o arquivo for importado por outro módulo, `main()` não é chamado automaticamente. Isso permite reutilizar as funções de `main.py` em outros scripts ou testes.

---

## Como usar

```bash
# Ativar o ambiente virtual (Windows)
.venv\Scripts\activate

# 8 rainhas com parâmetros padrão
python app/main.py

# 12 rainhas com população maior e verbose
python app/main.py -n 12 -p 100 -g 3000 -m 0.25 -v

# Ver todos os parâmetros disponíveis
python app/main.py --help
```

---

## Tabela de parâmetros

| Flag | Longa | Tipo | Padrão | Descrição |
|---|---|---|---|---|
| `-n` | `--queens` | int | 8 | Número de rainhas |
| `-p` | `--pop` | int | 50 | Tamanho da população |
| `-e` | `--elite` | int | 2 | Indivíduos preservados por elitismo |
| `-g` | `--generations` | int | 1000 | Máximo de gerações |
| `-m` | `--mutation` | float | 0.2 | Taxa de mutação (0.0 a 1.0) |
| `-v` | `--verbose` | flag | False | Mostrar fitness a cada geração |

---

## Saída esperada

```
==================================================
  Algoritmo Genético — 8-Rainhas
==================================================
  População: 50 | Elite: 2
  Max gerações: 1000 | Mutação: 20%
==================================================

Solução encontrada na geração 23!
Genes: [4, 0, 7, 5, 2, 6, 1, 3]

+---+---+---+---+---+---+---+---+
| . | . | . | . | Q | . | . | . |
+---+---+---+---+---+---+---+---+
| Q | . | . | . | . | . | . | . |
...
```
