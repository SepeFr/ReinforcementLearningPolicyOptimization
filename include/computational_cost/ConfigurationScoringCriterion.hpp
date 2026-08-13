#ifndef CONFIGURATION_SCORING_CRITERION_H
#define CONFIGURATION_SCORING_CRITERION_H

/** @addtogroup computational_cost_api
 * @{ */

#include <cstddef>
#include "BlackBoxProblem.hpp"
#include "ComputationCostUpperBounds.hpp"
#include "ConfigurationEvaluation.hpp"
#include "ConfigurationScoringConfiguration.hpp"
#include "ModelConfiguration.hpp"

/**
 * @brief Maps validation performance and execution metrics to a minimization score.
 *
 * The criterion adds the objective performance term to enabled normalized cost
 * penalties. Maximization performance is negated before the penalties are
 * added. Validation variance is checked and has coefficient zero in the score.
 * A normalized cost above one is accepted and reported through `std::clog`.
 *
 * @see ConfigurationEvaluation
 * @see cost_validation_chapter
 */
class ConfigurationScoringCriterion
{
  public:
  /**
   * @brief Creates a scoring criterion with fixed normalization bounds.
   * @param[in] configuration Unit costs, weights, and optional bound metadata.
   * @param[in] upper_bounds Positive simulation and neural-network denominators.
   * @param[in] maximum_parameter_count Architecture denominator used when its weight is positive.
   * @throws std::invalid_argument If a cost or weight is non-finite or negative,
   *         or if a denominator required by a positive weight is not positive and finite.
   */
  ConfigurationScoringCriterion( ConfigurationScoringConfiguration configuration,
                                 ComputationCostUpperBounds upper_bounds,
                                 std::size_t maximum_parameter_count );

  /**
   * @brief Computes the internal score used to rank a model configuration.
   *
   * With \f$\overline J_{\mathrm{val}}(c)\f$ equal to the validation mean
   * expressed in minimization direction, the score is
   * \f[
   * S(c) = \overline J_{\mathrm{val}}(c)
   *   + \eta_p\frac{P(c)}{P_{\max}}
   *   + \eta_s\frac{c_r E(c)+c_t T(c)}{C_{s,\max}}
   *   + \eta_n\frac{A(c)}{C_{n,\max}},
   * \f]
   * where the validation mean is sign-normalized from the supplied direction,
   * `P` is the network parameter count, `E` and `T` are episode and step
   * counters, and `A` is the neural-network MAC counter. Terms with zero weight
   * are omitted.
   *
   * @param[in] model_configuration Configuration whose architecture is measured.
   * @param[in] evaluation Validation statistics and accumulated execution metrics.
   * @param[in] direction Natural direction of the validation objective.
   * @return A finite score in which lower values rank first.
   * @throws std::invalid_argument If the validation mean is non-finite or the
   *         validation variance is non-finite or negative.
   * @throws std::overflow_error If deriving the network parameter count overflows.
   * @throws ConfigurationEvaluationError If the final selection score is non-finite.
   */
  double score( const ModelConfiguration &model_configuration,
                const ConfigurationEvaluation &evaluation,
                OptimizationDirection direction = OptimizationDirection::Minimize ) const;

  private:
  ConfigurationScoringConfiguration configuration_; ///< Immutable scoring coefficients after construction.
  ComputationCostUpperBounds upper_bounds_; ///< Denominators for measured execution costs.
  std::size_t maximum_parameter_count_; ///< Denominator for architecture complexity.
};

/** @} */

#endif // !CONFIGURATION_SCORING_CRITERION_H
