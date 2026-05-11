import random
from dataclasses import dataclass
from ga.individual import Individual
from ga.operators.selection import roulette_select
from ga.operators.crossover import single_point_crossover
from ga.operators.mutation import swap_mutation


@dataclass
class GAResult:
    solved: bool
    solution: Individual | None
    generation: int
    best_fitness_history: list[float]


def run(
    population: list[Individual],
    max_generations: int = 1000,
    elite_size: int = 2,
    mutation_rate: float = 0.1,
    verbose: bool = False,
    crossover_fn=None,
    mutation_fn=None,
    convergence_fn=None,
) -> GAResult:
    pop_size = len(population)
    best_fitness_history: list[float] = []

    if crossover_fn is None:
        crossover_fn = single_point_crossover
    if mutation_fn is None:
        mutation_fn = lambda ind: swap_mutation(ind) if random.random() < mutation_rate else ind
    if convergence_fn is None:
        convergence_fn = lambda ind: ind.fitness == 0

    for generation in range(1, max_generations + 1):
        # Ordena por fitness crescente (menor = melhor)
        population.sort(key=lambda ind: ind.fitness)
        best = population[0]
        best_fitness_history.append(best.fitness)

        if verbose:
            genes_str = ", ".join(f"{g:.4f}" for g in best.genes) if best.genes.dtype.kind == "f" else best.genes.tolist()
            print(f"Geração {generation:4d} | fitness: {best.fitness:.6f} | genes: [{genes_str}]")

        if convergence_fn(best):
            return GAResult(
                solved=True,
                solution=best.copy(),
                generation=generation,
                best_fitness_history=best_fitness_history,
            )

        # Elitismo: preserva os melhores
        elite = [ind.copy() for ind in population[:elite_size]]

        # Gera filhos para preencher o restante da população
        n_offspring = pop_size - elite_size
        offspring: list[Individual] = []

        while len(offspring) < n_offspring:
            parent_a, parent_b = roulette_select(population, 2)
            child_a, child_b = crossover_fn(parent_a, parent_b)
            mutation_fn(child_a)
            mutation_fn(child_b)

            offspring.append(child_a)
            if len(offspring) < n_offspring:
                offspring.append(child_b)

        population = elite + offspring

    # Não convergiu dentro do limite de gerações
    population.sort(key=lambda ind: ind.fitness)
    return GAResult(
        solved=False,
        solution=None,
        generation=max_generations,
        best_fitness_history=best_fitness_history,
    )
