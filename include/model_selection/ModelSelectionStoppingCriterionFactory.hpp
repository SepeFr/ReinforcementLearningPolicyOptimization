#ifndef MODEL_SELECTION_STOPPING_CRITERION_FACTORY_H
#define MODEL_SELECTION_STOPPING_CRITERION_FACTORY_H

/** @addtogroup model_selection_api
 * @{ */

#include <memory>
#include "ModelSelectionStoppingConfiguration.hpp"
#include "ModelSelectionStoppingStrategy.hpp"

/** @brief Instantiates a stopping strategy from its configuration variant. */
class ModelSelectionStoppingCriterionFactory
{
  public:
  /** @brief Prevents construction of this stateless factory. */
  ModelSelectionStoppingCriterionFactory() = delete;

  /**
   * @brief Creates the concrete strategy selected by a configuration alternative.
   * @param[in] configuration Budget, no-improvement, or target-score settings.
   * @return Unique ownership of the matching strategy.
   * @throws std::bad_variant_access If `configuration` is valueless.
   */
  static std::unique_ptr< ModelSelectionStoppingStrategy >
  create( const ModelSelectionStoppingConfiguration &configuration );
};

/** @} */

#endif // !MODEL_SELECTION_STOPPING_CRITERION_FACTORY_H
