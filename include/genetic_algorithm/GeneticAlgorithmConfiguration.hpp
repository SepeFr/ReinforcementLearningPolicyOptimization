#ifndef GENETIC_ALGORITHM_CONFIGURATION_H
#define GENETIC_ALGORITHM_CONFIGURATION_H

/**
 * @defgroup genetic_algorithm_api Genetic Algorithm
 * @brief Population representation, fitness, and evolutionary operators.
 * @{
 */

#include <cstddef>
#include <cstdint>

/** @brief Configures population evolution in GeneticAlgorithmEngine. */
struct GeneticAlgorithmConfiguration
{
  std::size_t population_size = 10; ///< Number of genomes per generation; must be at least two.
  std::size_t elite_count = 1; ///< Highest-fitness successful genomes copied unchanged; must be below population size.
  std::size_t tournament_size = 2; ///< Parent-selection draws with replacement; must be in `[1, population_size]`.
  std::uint64_t random_seed = 0; ///< Seed for tournament selection, crossover, and mutation randomness.
};

/** @} */

#endif // !GENETIC_ALGORITHM_CONFIGURATION_H
