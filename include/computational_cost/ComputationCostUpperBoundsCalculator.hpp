#ifndef COMPUTATION_COST_UPPER_BOUNDS_CALCULATOR_H
#define COMPUTATION_COST_UPPER_BOUNDS_CALCULATOR_H

/** @addtogroup computational_cost_api
 * @{ */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include "ComputationCostUpperBounds.hpp"
#include "ConfigurationScoringConfiguration.hpp"
#include "ValidationProtocolConfiguration.hpp"

/**
 * @brief Derives scoring denominators from evaluation and episode limits.
 *
 * Holdout and K-fold protocols produce different episode-count bounds. For a
 * holdout configuration the calculator uses
 * \f[
 * E_{\max}=B\,|S_{train}|\,R_{train}
 *          +|S_{validation}|\,R_{validation},
 * \f]
 * where `B` is `maximum_black_box_evaluations`. For K-fold validation it sums
 * the same training and validation contributions over the concrete fold sizes.
 * The simulation-step bound is \f$E_{\max}T_{\max}\f$. This bound is converted
 * to configured simulation cost units or multiplied by the maximum MAC count
 * per inference for the neural-network denominator.
 *
 * @see ConfigurationScoringCriterion
 * @see cost_validation_chapter
 */
class ComputationCostUpperBoundsCalculator
{
  public:
  /**
   * @brief Calculates the normalization bounds required by active score terms.
   *
   * Explicit bounds in `scoring_configuration` take precedence. Missing bounds
   * are derived only for components whose weight is positive.
   *
   * @tparam Scenario Scenario value stored by the validation configuration.
   * @tparam Environment Environment type constructed by the validation protocol.
   * @param[in] scoring_configuration Cost weights, unit costs, and optional explicit bounds.
   * @param[in] validation_configuration Holdout or K-fold scenario and replica configuration.
   * @param[in] inputs Maximum evaluation, episode-length, and inference-cost inputs.
   * @return Finite, positive simulation and neural-network denominators.
   * @throws std::invalid_argument If a required input is absent or zero; if
   *         replica counts, fold count, or scenario cardinalities cannot define
   *         the requested bound; or if a resulting bound is non-finite or not positive.
   */
  template< typename Scenario, typename Environment >
  static ComputationCostUpperBounds
  calculate( const ConfigurationScoringConfiguration &scoring_configuration,
             const ValidationConfiguration< Scenario, Environment > &validation_configuration,
             const ComputationCostUpperBoundInputs &inputs )
  {
    const bool calculate_simulation_bound = scoring_configuration.simulation_cost_weight > 0.0 &&
      !scoring_configuration.maximum_simulation_cost.has_value();
    const bool calculate_neural_network_bound = scoring_configuration.neural_network_cost_weight > 0.0 &&
      !scoring_configuration.maximum_neural_network_cost.has_value();

    ComputationCostUpperBounds upper_bounds;

    if ( scoring_configuration.simulation_cost_weight > 0.0 &&
         scoring_configuration.maximum_simulation_cost.has_value() )
    {
      upper_bounds.maximum_simulation_cost = *scoring_configuration.maximum_simulation_cost;
    }

    if ( scoring_configuration.neural_network_cost_weight > 0.0 &&
         scoring_configuration.maximum_neural_network_cost.has_value() )
    {
      upper_bounds.maximum_neural_network_cost = *scoring_configuration.maximum_neural_network_cost;
    }

    if ( calculate_simulation_bound || calculate_neural_network_bound )
    {
      if ( !inputs.maximum_black_box_evaluations.has_value() || *inputs.maximum_black_box_evaluations == 0 )
      {
        throw std::invalid_argument(
          "ComputationCostUpperBoundsCalculator: maximum black-box evaluations must be provided and positive" );
      }
      if ( !inputs.maximum_episode_steps.has_value() || *inputs.maximum_episode_steps == 0 )
      {
        throw std::invalid_argument(
          "ComputationCostUpperBoundsCalculator: maximum episode steps must be provided and positive" );
      }

      const double maximum_evaluation_count = static_cast< double >( *inputs.maximum_black_box_evaluations );
      const double maximum_episode_count = std::visit(
        [maximum_evaluation_count]( const auto &configuration )
        {
          using Configuration = std::remove_cvref_t< decltype( configuration ) >;

          if ( configuration.training_runs_per_scenario == 0 ||
               configuration.validation_runs_per_scenario == 0 )
          {
            throw std::invalid_argument(
              "ComputationCostUpperBoundsCalculator: runs per scenario must be positive" );
          }

          const double training_runs = static_cast< double >( configuration.training_runs_per_scenario );
          const double validation_runs = static_cast< double >( configuration.validation_runs_per_scenario );

          if constexpr ( std::is_same_v< Configuration,
                                         KFoldCrossValidationConfiguration< Scenario, Environment > > )
          {
            if ( configuration.fold_count < 2 )
            {
              throw std::invalid_argument(
                "ComputationCostUpperBoundsCalculator: fold count must be at least two" );
            }

            const std::size_t scenario_count =
              configuration.training_scenarios.size() + configuration.validation_scenarios.size();
            if ( scenario_count < configuration.fold_count )
            {
              throw std::invalid_argument(
                "ComputationCostUpperBoundsCalculator: scenario count must be at least the fold count" );
            }

            const std::size_t base_fold_size = scenario_count / configuration.fold_count;
            const std::size_t remaining_scenarios = scenario_count % configuration.fold_count;
            double episode_count = 0.0;

            for ( std::size_t fold = 0; fold < configuration.fold_count; ++fold )
            {
              const std::size_t validation_size = base_fold_size + ( fold < remaining_scenarios ? 1 : 0 );
              const std::size_t training_size = scenario_count - validation_size;
              episode_count += maximum_evaluation_count * static_cast< double >( training_size ) * training_runs;
              episode_count += static_cast< double >( validation_size ) * validation_runs;
            }
            return episode_count;
          }
          else
          {
            if ( configuration.training_scenarios.empty() || configuration.validation_scenarios.empty() )
            {
              throw std::invalid_argument(
                "ComputationCostUpperBoundsCalculator: training and validation scenarios cannot be empty" );
            }

            return maximum_evaluation_count * static_cast< double >( configuration.training_scenarios.size() ) *
                training_runs +
              static_cast< double >( configuration.validation_scenarios.size() ) * validation_runs;
          }
        },
        validation_configuration );

      const double maximum_simulation_steps =
        maximum_episode_count * static_cast< double >( *inputs.maximum_episode_steps );

      if ( calculate_simulation_bound )
      {
        upper_bounds.maximum_simulation_cost =
          scoring_configuration.reset_cost * maximum_episode_count +
          scoring_configuration.simulation_step_cost * maximum_simulation_steps;
      }

      if ( calculate_neural_network_bound )
      {
        if ( !inputs.maximum_macs_per_inference.has_value() || *inputs.maximum_macs_per_inference == 0 )
        {
          throw std::invalid_argument(
            "ComputationCostUpperBoundsCalculator: maximum MACs per inference must be provided and positive" );
        }
        upper_bounds.maximum_neural_network_cost =
          maximum_simulation_steps * static_cast< double >( *inputs.maximum_macs_per_inference );
      }
    }

    if ( !std::isfinite( upper_bounds.maximum_simulation_cost ) ||
         upper_bounds.maximum_simulation_cost <= 0.0 )
    {
      throw std::invalid_argument(
        "ComputationCostUpperBoundsCalculator: maximum simulation cost must be finite and positive" );
    }
    if ( !std::isfinite( upper_bounds.maximum_neural_network_cost ) ||
         upper_bounds.maximum_neural_network_cost <= 0.0 )
    {
      throw std::invalid_argument(
        "ComputationCostUpperBoundsCalculator: maximum neural network cost must be finite and positive" );
    }

    return upper_bounds;
  }

  private:
  /** @brief Prevents construction of this stateless utility class. */
  ComputationCostUpperBoundsCalculator() = delete;
};

/** @} */

#endif // !COMPUTATION_COST_UPPER_BOUNDS_CALCULATOR_H
