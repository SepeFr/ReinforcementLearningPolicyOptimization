#ifndef MODEL_CONFIGURATION_H
#define MODEL_CONFIGURATION_H

/** @addtogroup model_selection_api
 * @{ */

#include "OptimizerConfiguration.hpp"
#include "PolicyConfiguration.hpp"

/**
 * @brief Couples the optimizer and policy settings evaluated by model selection.
 *
 * Search-space axes normally point into fields nested in this aggregate. The
 * aggregate that owns those fields must remain at a stable address while such
 * axes or selectors exist.
 */
struct ModelConfiguration
{
  public:
  OptimizerConfiguration optimizer_configuration; ///< Training optimizer, initialization, method, and stopping settings.
  PolicyConfiguration policy; ///< Policy architecture and objective regularization settings.
};

/** @} */

#endif // !MODEL_CONFIGURATION_H
