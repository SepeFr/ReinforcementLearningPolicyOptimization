#ifndef GENETIC_INDIVIDUAL_H
#define GENETIC_INDIVIDUAL_H

/** @addtogroup genetic_algorithm_api
 * @{ */

/** @file GeneticIndividual.hpp @brief Genetic evaluation status, fitness, and individual value types. */

#include <optional>

/** @brief Identifies whether an individual's fitness is usable for reproduction. */
enum class GeneticEvaluationStatus
{
  Succeeded, ///< The fitness value may participate in elitism and parent selection.
  Failed ///< The individual is excluded from elitism and parent selection.
};

/** @brief Stores a non-negative reproduction weight and its evaluation status. */
struct GeneticFitness
{
  double value = 0.0; ///< Finite weight for a successful individual; negative input is clamped to zero.
  GeneticEvaluationStatus status = GeneticEvaluationStatus::Failed; ///< Evaluation outcome controlling eligibility.
};

/**
 * @brief Couples a genome with optional feedback for its current generation.
 * @tparam Genome Value type consumed by crossover and mutation operators.
 */
template< typename Genome >
struct GeneticIndividual
{
  Genome genome; ///< Candidate representation.
  std::optional< GeneticFitness > fitness; ///< Empty before feedback for the current generation.
};

/** @} */

#endif // !GENETIC_INDIVIDUAL_H
