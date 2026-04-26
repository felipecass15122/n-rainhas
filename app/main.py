import argparse
import sys
import os

# Garante que o diretório app/ está no path para imports relativos funcionarem
sys.path.insert(0, os.path.dirname(__file__))

from ga.population import create_population
from ga.engine import run
from utils.visualizer import print_board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algoritmo Genético para o Problema das N-Rainhas"
    )
    parser.add_argument(
        "-n", "--queens",
        type=int,
        default=8,
        metavar="N",
        help="Número de rainhas (padrão: 8)",
    )
    parser.add_argument(
        "-p", "--pop",
        type=int,
        default=50,
        metavar="POP",
        help="Tamanho da população (padrão: 20)",
    )
    parser.add_argument(
        "-e", "--elite",
        type=int,
        default=2,
        metavar="ELITE",
        help="Número de indivíduos preservados por elitismo (padrão: 2)",
    )
    parser.add_argument(
        "-g", "--generations",
        type=int,
        default=1000,
        metavar="GENS",
        help="Máximo de gerações (padrão: 1000)",
    )
    parser.add_argument(
        "-m", "--mutation",
        type=float,
        default=0.2,
        metavar="RATE",
        help="Taxa de mutação entre 0.0 e 1.0 (padrão: 0.1)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Exibe o progresso a cada geração",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  Algoritmo Genético — {args.queens}-Rainhas")
    print(f"{'='*50}")
    print(f"  População: {args.pop} | Elite: {args.elite}")
    print(f"  Max gerações: {args.generations} | Mutação: {args.mutation:.0%}")
    print(f"{'='*50}\n")

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

    print()


if __name__ == "__main__":
    main()
