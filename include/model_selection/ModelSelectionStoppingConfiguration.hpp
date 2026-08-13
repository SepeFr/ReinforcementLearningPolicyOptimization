#ifndef MODEL_SELECTION_STOPPING_CONFIGURATION_H
#define MODEL_SELECTION_STOPPING_CONFIGURATION_H

/** @addtogroup model_selection_api
 * @{ */

/** @file ModelSelectionStoppingConfiguration.hpp @brief Model-selection stopping configuration alternatives. */

#include <cstddef>
#include <variant>

struct ModelSelectionBudgetCriterionConfiguration;
struct ModelSelectionNoImprovementCriterionConfiguration;
struct ModelSelectionTargetScoreCriterionConfiguration;

/** @brief Selects one model-selection stopping strategy configuration. */
using ModelSelectionStoppingConfiguration =
  std::variant< ModelSelectionBudgetCriterionConfiguration, ModelSelectionNoImprovementCriterionConfiguration,
                ModelSelectionTargetScoreCriterionConfiguration >;

/** @brief Stops after a fixed number of attempted configurations. */
struct ModelSelectionBudgetCriterionConfiguration
{
  std::size_t maximum_iterations; ///< Positive attempt budget; failed and invalid configurations consume one iteration.
};

/** @brief Stops after a run of successful evaluations without sufficient score improvement. */
struct ModelSelectionNoImprovementCriterionConfiguration
{
  std::size_t maximum_iterations_without_improvement; ///< Positive number of non-improving successful evaluations allowed.
  double threshold; ///< Finite non-negative strict decrease required to reset the counter.
};

/** @brief Stops when a successful lower-is-better selection score reaches a target. */
struct ModelSelectionTargetScoreCriterionConfiguration
{
  double target_score; ///< Finite inclusive upper bound for a successful selection score.
};

/** @} */

#endif // !MODEL_SELECTION_STOPPING_CONFIGURATION_H
