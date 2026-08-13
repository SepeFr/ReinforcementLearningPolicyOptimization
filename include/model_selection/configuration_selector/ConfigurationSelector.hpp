#ifndef CONFIGURATION_SELECTOR_H
#define CONFIGURATION_SELECTOR_H

/** @addtogroup model_selection_api
 * @{ */

/** @file ConfigurationSelector.hpp @brief Configuration ask--tell interface and feedback types. */

#include <cstddef>
#include <vector>
#include "ConfigurationEvaluation.hpp"
#include "ModelConfiguration.hpp"

/** @brief Reports whether one requested configuration produced usable feedback. */
enum class ConfigurationEvaluationStatus
{
  Succeeded, ///< `evaluation` contains a valid selection score.
  Failed ///< The configuration could not be evaluated or scored.
};

/** @brief Associates one selector request with its evaluation outcome. */
struct ConfigurationFeedback
{
  ConfigurationEvaluation evaluation; ///< Statistics and score; meaningful to adaptive selectors on success.
  ConfigurationEvaluationStatus status = ConfigurationEvaluationStatus::Failed; ///< Outcome of the matching requested configuration.
};

/**
 * @brief Defines the ask--tell protocol for model-configuration generation.
 *
 * Returned configuration order defines feedback association. Integrated
 * selectors retain non-owning references to a ModelConfiguration and its axis
 * fields. That configuration must remain alive at a stable address for the
 * selector lifetime.
 *
 * @see model_selection_chapter
 */
class ConfigurationSelector
{
  public:
  /** @brief Destroys a selector through its interface. */
  virtual ~ConfigurationSelector() = default;

  /**
   * @brief Reports whether an external stopping budget is required.
   * @return `true` by default for selectors with an unbounded generation stream.
   */
  virtual bool requiresStoppingBudget() const noexcept { return true; }
  /** @brief Reports whether another configuration can be requested. @return Availability of at least one next item. */
  virtual bool hasNext() const = 0;
  /**
   * @brief Requests the next ordered batch of model configurations.
   * @param[in] maximum_batch_size Positive upper limit for the returned cardinality.
   * @return Configurations in the order expected by the matching `tell()` call.
   */
  virtual std::vector< ModelConfiguration > ask( std::size_t maximum_batch_size ) = 0;
  /**
   * @brief Supplies ordered outcomes for the preceding adaptive request.
   * @param[in] feedback Outcomes in the same order as the pending batch.
   */
  virtual void tell( const std::vector< ConfigurationFeedback > &feedback ) = 0;
};

/** @} */

#endif // !CONFIGURATION_SELECTOR_H
