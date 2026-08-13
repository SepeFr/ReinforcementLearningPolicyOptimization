#ifndef CONFIGURATION_EVALUATION_H
#define CONFIGURATION_EVALUATION_H

/** @addtogroup model_selection_api
 * @{ */

#include <optional>
#include "ExecutionMetrics.hpp"

/**
 * @brief Stores validation statistics, measured work, and an optional ranking score.
 *
 * Validation statistics retain the objective's natural direction. Model
 * selection assigns `selection_score` after validation and uses a lower-is-better
 * convention for all selectors and stopping criteria.
 */
struct ConfigurationEvaluation
{
  double validation_mean = 0.0; ///< Arithmetic mean of validation samples in the objective's natural direction.
  double validation_variance = 0.0; ///< Population variance of validation samples; finite and non-negative on success.
  ExecutionMetrics execution_metrics; ///< Training and validation work attributed to this configuration.
  std::optional< double > selection_score; ///< Finite lower-is-better score after ConfigurationScoringCriterion succeeds.
};

/** @} */

#endif // !CONFIGURATION_EVALUATION_H
