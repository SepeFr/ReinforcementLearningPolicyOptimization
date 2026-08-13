#ifndef STOPPING_CRITERION_FACTORY_H
#define STOPPING_CRITERION_FACTORY_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <memory>
#include "BlackBoxProblem.hpp"
#include "OptimizationStoppingStrategy.hpp"
#include "StoppingConfiguration.hpp"

/** @brief Creates a concrete stopping strategy from variant configuration. */
class StoppingCriterionFactory
{
  public:
  /** @brief Static utility class; instances cannot be constructed. */
  StoppingCriterionFactory() = delete;

  /**
   * @brief Creates one owned stopping strategy.
   * @param[in] configuration Concrete strategy settings.
   * @param[in] direction Natural problem direction; a maximizing target is negated.
   * @return Unique ownership of the matching strategy.
   */
  static std::unique_ptr< OptimizationStoppingStrategy >
  create( const StoppingConfiguration &configuration, OptimizationDirection direction );
};

/** @} */

#endif // !STOPPING_CRITERION_FACTORY_H
