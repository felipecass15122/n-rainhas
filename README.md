# N-Rainhas com Algoritmo Genético

Solução para o **Problema das N-Rainhas** usando um Algoritmo Genético (AG) implementado do zero em Python.

O problema consiste em posicionar N rainhas em um tabuleiro N×N de forma que nenhuma rainha ataque outra — sem conflitos em linhas, colunas ou diagonais.

## Como funciona

O AG usa **codificação por permutação**: o gene `i` representa a coluna da rainha na linha `i`. Essa representação elimina conflitos de linha e coluna por construção, reduzindo o espaço de busca.

| Componente       | Estratégia                                              |
|------------------|---------------------------------------------------------|
| Fitness          | Contagem de conflitos diagonais (objetivo: 0)           |
| Seleção          | Roleta (peso inversamente proporcional ao fitness)      |
| Crossover        | Ponto único com reparo para manter permutação válida    |
| Mutação          | Troca de dois genes aleatórios                          |
| Elitismo         | Os melhores indivíduos são preservados entre gerações   |

## Estrutura do projeto

```
app/
├── main.py               # Ponto de entrada (CLI)
├── ga/                   # Framework genérico de AG
│   ├── engine.py         # Loop principal do AG
│   ├── individual.py     # Classe base abstrata
│   ├── population.py     # Geração de população
│   └── operators/        # Seleção, crossover e mutação
└── problems/nqueens/     # Implementação específica do N-Rainhas
    ├── individual.py
    └── fitness.py
tests/                    # Testes automatizados (pytest)
docs/                     # Documentação detalhada de cada módulo
```

## Pré-requisitos

- Python 3.10+

## Instalação

```bash
# Clone o repositório
git clone <url-do-repo>
cd n-rainhas

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Instale as dependências
pip install -r requirements.txt
```

## Executando

```bash
python app/main.py [opções]
```

### Opções disponíveis

| Flag               | Descrição                          | Padrão |
|--------------------|------------------------------------|--------|
| `-n, --queens N`   | Número de rainhas                  | `8`    |
| `-p, --pop N`      | Tamanho da população               | `50`   |
| `-e, --elite N`    | Tamanho do elitismo                | `2`    |
| `-g, --generations N` | Máximo de gerações              | `1000` |
| `-m, --mutation F` | Taxa de mutação (0.0 a 1.0)        | `0.2`  |
| `-v, --verbose`    | Exibe progresso geração a geração  | —      |

### Exemplos

```bash
# 8 rainhas com configuração padrão
python app/main.py

# 12 rainhas com população maior e saída detalhada
python app/main.py -n 12 -p 100 -g 2000 -v

# 4 rainhas para teste rápido
python app/main.py -n 4 -v
```

### Saída esperada

```
==================================================
  Algoritmo Genético — 8-Rainhas
==================================================
  População: 50 | Elite: 2
  Max gerações: 1000 | Mutação: 20%
==================================================

Solução encontrada na geração 42!
Genes: [3, 6, 2, 7, 1, 4, 0, 5]

+---+---+---+---+---+---+---+---+
| . | . | . | Q | . | . | . | . |
+---+---+---+---+---+---+---+---+
| . | . | . | . | . | . | Q | . |
+---+---+---+---+---+---+---+---+
...
```

## Testes

```bash
pytest
```

Os testes cobrem a função de fitness, geração de população válida e convergência do AG para N=4 e N=8.
