#ifndef GENETIC_CONFIGURATION_SELECTOR_CONFIGURATION_H
#define GENETIC_CONFIGURATION_SELECTOR_CONFIGURATION_H

/** @addtogroup model_selection_api
 * @{ */

/** @file GeneticConfigurationSelectorConfiguration.hpp @brief Genetic configuration-search operator alternatives. */

#include <variant>
#include "GeneticAlgorithmConfiguration.hpp"

/** @brief Configures independent uniform parent selection for every genome axis. */
struct UniformCrossoverConfiguration
{
  double first_parent_gene_probability = 0.5; ///< Finite probability in `[0,1]` of copying each gene from the first parent.
};

/** @brief Configures independent random-reset mutation for every genome axis. */
struct RandomResetMutationConfiguration
{
  double per_axis_probability = 0.1; ///< Finite probability in `[0,1]` of resampling each axis from its full domain.
};

/** @brief Selects the crossover operator used for configuration genomes. */
using ConfigurationCrossoverConfiguration = std::variant< UniformCrossoverConfiguration >;
/** @brief Selects the mutation operator used for configuration genomes. */
using ConfigurationMutationConfiguration = std::variant< RandomResetMutationConfiguration >;

/** @brief Configures genetic population evolution and score-to-fitness mapping. */
struct GeneticConfigurationSelectorConfiguration
{
  GeneticAlgorithmConfiguration algorithm; ///< Population size, selection, elitism, and random seed.
  ConfigurationCrossoverConfiguration crossover = UniformCrossoverConfiguration{}; ///< Per-gene parent selection rule.
  ConfigurationMutationConfiguration mutation = RandomResetMutationConfiguration{}; ///< Per-axis replacement rule.
  double fitness_epsilon = 1e-12; ///< Finite positive offset that gives every successful individual positive fitness.
};

/** @} */

#endif // !GENETIC_CONFIGURATION_SELECTOR_CONFIGURATION_H
