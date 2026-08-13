#ifndef MODEL_SELECTION_STOPPING_CRITERION_H
#define MODEL_SELECTION_STOPPING_CRITERION_H

/** @addtogroup model_selection_api
 * @{ */

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include "ConfigurationSelector.hpp"
#include "ModelSelectionStoppingConfiguration.hpp"
#include "ModelSelectionStoppingStrategy.hpp"
#include "OptimizationTerminationReason.hpp"

/**
 * @brief Owns and evaluates a heterogeneous ordered set of stopping strategies.
 *
 * At most one strategy of each variant alternative is stored. Strategies are
 * visited in configuration order. The first triggered strategy ends processing
 * for that feedback item and supplies the termination reason.
 *
 * @see ModelSelection
 * @see model_selection_chapter
 */
class ModelSelectionStoppingCriterion
{
  public:
  /**
   * @brief Creates a composite containing one strategy.
   * @param[in] configuration Strategy alternative to instantiate.
   * @throws std::out_of_range If the variant is valueless.
   */
  explicit ModelSelectionStoppingCriterion( ModelSelectionStoppingConfiguration configuration );

  /**
   * @brief Creates strategies in the supplied evaluation order.
   * @param[in] configurations Distinct strategy alternatives.
   * @throws std::invalid_argument If an alternative type occurs more than once.
   * @throws std::out_of_range If an input variant is valueless.
   */
  explicit ModelSelectionStoppingCriterion( const std::vector< ModelSelectionStoppingConfiguration > &configurations );

  /**
   * @brief Updates strategies until one requests termination.
   * @param[in] current_feedback Outcome of one attempted configuration.
   * @return First triggered reason, or an empty optional when execution should continue.
   */
  std::optional< OptimizationTerminationReason > stoppingReason( const ConfigurationFeedback &current_feedback );

  private:
  /** @brief Fixed slots indexed by `ModelSelectionStoppingConfiguration::index()`. */
  using CriterionSlots = std::array< std::unique_ptr< ModelSelectionStoppingStrategy >,
                                     std::variant_size_v< ModelSelectionStoppingConfiguration > >;

  /**
   * @brief Creates and inserts one strategy while preserving caller order.
   * @param[in] configuration Strategy alternative to add.
   * @throws std::invalid_argument If its alternative type is already present.
   * @throws std::out_of_range If the variant is valueless.
   */
  void addCriterion( const ModelSelectionStoppingConfiguration &configuration );

  CriterionSlots criteria_{}; ///< Owned strategy by variant alternative index.
  std::vector< std::size_t > criterion_order_; ///< Alternative indices in caller-supplied evaluation order.
};

/** @} */

#endif // !MODEL_SELECTION_STOPPING_CRITERION_H
