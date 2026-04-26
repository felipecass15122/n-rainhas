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
    best_fitness_history: list[int]


def run(
    population: list[Individual],
    max_generations: int = 1000,
    elite_size: int = 2,
    mutation_rate: float = 0.1,
    verbose: bool = False,
) -> GAResult:
    pop_size = len(population)
    best_fitness_history: list[int] = []

    for generation in range(1, max_generations + 1):
        # Ordena por fitness crescente (menor = melhor)
        population.sort(key=lambda ind: ind.fitness)
        best = population[0]
        best_fitness_history.append(best.fitness)

        if verbose:
            print(f"Geração {generation:4d} | melhor fitness: {best.fitness}")

        if best.fitness == 0:
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
            child_a, child_b = single_point_crossover(parent_a, parent_b)

            if random.random() < mutation_rate:
                swap_mutation(child_a)
            if random.random() < mutation_rate:
                swap_mutation(child_b)

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
