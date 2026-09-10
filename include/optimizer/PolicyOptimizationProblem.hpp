#ifndef POLICY_OPTIMIZATION_PROBLEM_H
#define POLICY_OPTIMIZATION_PROBLEM_H

/** @addtogroup optimizer_core_api
 * @{ */

/** @file PolicyOptimizationProblem.hpp @brief Policy training objective over scenarios and episode replicas. */

#include <cstddef>
#include <cstdint>
#include <eigen3/Eigen/Core>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "BlackBoxProblem.hpp"
#include "EpisodeResult.hpp"
#include "EpisodeRunner.hpp"
#include "ExecutionMetrics.hpp"
#include "FeedForwardNetworkComputationCost.hpp"
#include "FeedForwardNetworkConfiguration.hpp"
#include "ObjectiveEvaluation.hpp"
#include "ParametrizedPolicy.hpp"
#include "PolicyConfiguration.hpp"
#include "PolicyFactory.hpp"

/** @brief Controls whether policy weight regularization contributes to episode samples. */
enum class RegularizationApplication
{
  Include, ///< Adds the configured negative weight penalty to every episode return.
  Exclude  ///< Uses episode returns directly.
};

namespace
{
  /**
   * @brief Computes the maximization-space penalty for a policy's weights.
   * @param[in] weights Canonically ordered weights; bias values are excluded.
   * @param[in] scale Non-negative regularization coefficient.
   * @param[in] type Selected L1 or squared-L2 penalty.
   * @return `-scale * weights.lpNorm<1>()` for L1, or
   * `-scale * weights.squaredNorm()` for L2.
   * @throws std::invalid_argument If @p type is unsupported.
   */
  double calculateRegularization( Eigen::Ref< const Eigen::VectorXd > weights, double scale, RegularizationType type )
  {
    switch ( type )
    {
      case RegularizationType::L1_Regularization:
        return -scale * weights.cwiseAbs().sum();
      case RegularizationType::L2_Regularization:
        return -scale * weights.squaredNorm();
    }

    throw std::invalid_argument( "PolicyOptimizationProblem: unsupported regularization type" );
  }
}; // namespace

/**
 * @brief Evaluates policy parameters through repeated simulated episodes.
 *
 * Scenarios are processed in stored order. Each scenario contributes
 * number_of_runs consecutive samples, giving
 * `scenarios.size() * number_of_runs` samples on a complete evaluation. The
 * The configured optimization direction determines how episode returns are interpreted.
 *
 * @tparam ScenarioType Scenario accepted by the environment.
 * @tparam ObservationType Observation shared by environment and policy.
 * @tparam ActionType Action shared by policy and environment.
 * @tparam EnvironmentType Concrete stateful environment type.
 * @tparam RunnerType Episode runner implementation used for every replica.
 * @see simulation_execution_chapter
 * @see optimization_lifecycle_chapter
 */
template< typename ScenarioType, typename ObservationType, typename ActionType, typename EnvironmentType,
          typename RunnerType = EpisodeRunner< EnvironmentType, ParametrizedPolicy< ObservationType, ActionType >,
                                               EpisodeResult< typename EnvironmentType::TerminationReason > > >
class PolicyOptimizationProblem : public BlackBoxProblem
{
  static_assert( std::is_same_v< typename EnvironmentType::Observation, ObservationType >,
                 "ObservationType and Environment::ObservationType differs" );
  static_assert( std::is_same_v< typename EnvironmentType::Action, ActionType >,
                 "ActionType and Environment::ActionType differs" );

  public:
  using Scenario = ScenarioType;                            ///< Scenario stored and passed to Environment::reset().
  using Observation = ObservationType;                      ///< Environment-to-policy observation type.
  using Action = ActionType;                                ///< Policy-to-environment action type.
  using Environment = EnvironmentType;                      ///< Concrete environment type.
  using Policy = ParametrizedPolicy< Observation, Action >; ///< Runtime policy interface.
  using Result = EpisodeResult< typename Environment::TerminationReason >; ///< Episode aggregate type.
  using Runner = RunnerType;                                               ///< Episode runner selected by the protocol.

  static_assert( std::is_base_of_v< EpisodeRunner< Environment, Policy, Result >, Runner >,
                 "RunnerType must derive from EpisodeRunner with matching types" );

  /** @brief Destroys the owned policy and regularization network. */
  ~PolicyOptimizationProblem() = default;
  /**
   * @brief Builds evaluation components and retains a non-owning environment reference.
   * @param[in] configuration Policy structure and regularization copied into the problem.
   * @param[in] scenarios Nonempty scenario sequence moved into evaluation order.
   * @param[in] number_of_runs Positive replica count per scenario.
   * @param[in,out] environment Environment reused sequentially by every replica.
   * It must outlive this problem and every call to evaluate().
   * @param[in] regularization_application Whether to add the configured weight penalty.
   * @param[in] direction Whether episode returns are minimized or maximized.
   * @throws std::invalid_argument If @p scenarios is empty, @p number_of_runs
   * is zero, or the network configuration is invalid.
   * @throws InvalidConfigurationError If PolicyFactory rejects @p configuration.
   * @throws std::overflow_error If network dimensions or cost counts exceed supported ranges.
   */
  PolicyOptimizationProblem(
    PolicyConfiguration configuration, std::vector< Scenario > scenarios, std::size_t number_of_runs,
    Environment &environment,
    RegularizationApplication regularization_application = RegularizationApplication::Include,
    OptimizationDirection direction = OptimizationDirection::Maximize ) :
      BlackBoxProblem( FeedForwardNetworkConfiguration::parameterCountFromConfiguration( configuration.network ),
                       direction ),
      configuration_( std::move( configuration ) ),
      policy_( PolicyFactory::create< Observation, Action >( configuration_ ) ),
      regularization_network_( PolicyFactory::create( configuration_ ) ), scenarios_( std::move( scenarios ) ),
      number_of_runs_( number_of_runs ),
      macs_per_inference_( FeedForwardNetworkComputationCost::macsPerInference( configuration_.network ) ),
      regularization_application_( regularization_application ), environment_( environment )
  {
    if ( scenarios_.empty() )
    {
      throw std::invalid_argument( "PolicyOptimizationProblem: at least one scenario is required" );
    }
    if ( number_of_runs_ == 0 )
    {
      throw std::invalid_argument( "PolicyOptimizationProblem: number of runs must be greater than zero" );
    }
  } 



  /**
   * @brief Runs every configured scenario and replica with one parameter vector.
   *
   * With regularization included, each sample is
   * \f[
   * R_{s,r} - \lambda\lVert w\rVert_1
   * \quad\text{or}\quad
   * R_{s,r} - \lambda\lVert w\rVert_2^2,
   * \f]
   * where `regularization_coefficient` is \f$\lambda\f$ and `w` contains
   * FeedForwardNetwork::weightsParameters() only. A final reason equal to the
   * environment's `Failure` value marks that sample failed while retaining its
   * finite value. Optimizer still accepts a finite aggregate independently of
   * its failed-sample count.
   *
   * @param[in] parameters Canonical policy parameter vector.
   * @return Episode samples in scenario-major, replica-minor order, with total
   * episode, step, inference, and MAC metrics.
   * @throws std::invalid_argument If the parameter length or an encoded observation dimension is invalid.
   * @note An std::overflow_error during an episode appends a failed NaN sample
   * and returns the partial aggregate immediately. Other environment and policy
   * exceptions propagate.
   */
  ObjectiveEvaluation evaluate( Eigen::Ref< const Eigen::VectorXd > parameters ) override
  {
    policy_->setParameters( parameters );
    double regularization = 0.0;
    if ( regularization_application_ == RegularizationApplication::Include &&
         configuration_.regularization_coefficient != 0.0 )
    {
      regularization_network_->setParameters( parameters );
      regularization =
        calculateRegularization( regularization_network_->weightsParameters(),
                                 configuration_.regularization_coefficient, configuration_.regularization_strategy );
    }

    ObjectiveEvaluation objective_evaluation;

    for ( const Scenario &scenario : scenarios_ )
    {
      for ( std::size_t run = 0; run < number_of_runs_; run++ )
      {
        using TerminationReason = typename Environment::TerminationReason;
        try
        {
          Runner episode_runner( environment_, *policy_, scenario );
          const Result result = episode_runner.run();
          const bool failed = result.termination_reason == TerminationReason::Failure;
          if ( !std::in_range< std::uint64_t >( result.steps ) )
          {
            throw std::overflow_error( "PolicyOptimizationProblem: episode step count is not representable" );
          }
          const std::uint64_t episode_steps = static_cast< std::uint64_t >( result.steps );

          ExecutionMetrics episode_metrics;
          episode_metrics.addEpisode( episode_steps );
          episode_metrics.addPolicyInferences( episode_steps, macs_per_inference_ );
          objective_evaluation.appendExecutionMetrics( episode_metrics );
          objective_evaluation.appendSample( result.total_reward + regularization, failed );
        }
        catch ( const std::overflow_error & )
        {
          objective_evaluation.appendSample( std::numeric_limits< double >::quiet_NaN(), true );
          return objective_evaluation;
        }
      }
    }
    return objective_evaluation;
  }


  protected:
  /** Copied policy structure and regularization settings. */
  PolicyConfiguration configuration_;
  /** Owned policy used for episode actions. */
  std::unique_ptr< Policy > policy_;
  /** Owned network used to separate weight parameters from biases. */
  std::unique_ptr< FeedForwardNetwork > regularization_network_;
  /** Owned scenarios in evaluation order. */
  std::vector< Scenario > scenarios_;
  /** Replica count executed for each scenario. */
  std::size_t number_of_runs_;
  /** Multiply-accumulate operations attributed to one policy inference. */
  std::uint64_t macs_per_inference_;
  /** Whether evaluate() adds the weight penalty. */
  RegularizationApplication regularization_application_;
  /** Non-owning environment reused across sequential episodes. */
  Environment &environment_;
};

/** @} */

#endif // !POLICY_OPTIMIZATION_PROBLEM_H
