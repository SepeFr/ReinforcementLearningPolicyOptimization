#ifndef OPTIMIZER_CONFIGURATION_H
#define OPTIMIZER_CONFIGURATION_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <cstddef>
#include <vector>
#include "InitializationConfiguration.hpp"
#include "MethodHyperparameters.hpp"
#include "StoppingConfiguration.hpp"

/** @brief Complete method, initialization, stopping, and reproducibility settings. */
struct OptimizerConfiguration
{
  MethodHyperparameters hyperparameters; ///< Concrete method and its algorithm-specific values.
  InitializationConfiguration initialization; ///< Initial-candidate strategy used by the factory.
  std::vector< StoppingConfiguration > stopping; ///< External criteria tested after each evaluated batch.
  std::size_t random_seed; ///< Seed forwarded to stochastic components created by OptimizerFactory.
};

/** @} */

#endif // !OPTIMIZER_CONFIGURATION_H
