import argparse
import sys
import os

_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _app_dir)
sys.path.insert(0, os.path.dirname(_app_dir))

from ga.population import create_rastrigin_population
from ga.operators.crossover import blx_alpha_crossover
from ga.operators.mutation import gaussian_mutation
from ga.engine import run


def main():
    parser = argparse.ArgumentParser(description="Algoritmo Genético para Otimização da Função Rastrigin")
    parser.add_argument("-d", "--dims",    type=int,   default=20,   metavar="D",    help="Dimensões (padrão: 20)")
    parser.add_argument("-p", "--pop",     type=int,   default=200,  metavar="POP",  help="Tamanho da população (padrão: 200)")
    parser.add_argument("-g", "--gens",    type=int,   default=2000, metavar="GENS", help="Máximo de gerações (padrão: 2000)")
    parser.add_argument("-e", "--elite",   type=int,   default=4,    metavar="ELITE",help="Tamanho do elitismo (padrão: 4)")
    parser.add_argument("-s", "--sigma",   type=float, default=0.1,  metavar="S",    help="Desvio padrão da mutação gaussiana (padrão: 0.1)")
    parser.add_argument("-t", "--tol",     type=float, default=1e-6, metavar="TOL",  help="Tolerância para convergência (padrão: 1e-6)")
    parser.add_argument("-v", "--verbose", action="store_true",                      help="Exibe progresso por geração")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Algoritmo Genético — Rastrigin {args.dims}D")
    print(f"{'='*50}")
    print(f"  População: {args.pop} | Elite: {args.elite}")
    print(f"  Max gerações: {args.gens} | sigma: {args.sigma} | tol: {args.tol}")
    print(f"{'='*50}\n")

    population = create_rastrigin_population(args.dims, args.pop)

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
