#ifndef EPISODE_RUNNER_H
#define EPISODE_RUNNER_H

/** @addtogroup simulation_api
 * @{ */

#include <exception>
#include <type_traits>
#include "EpisodeResult.hpp"

/**
 * @brief Executes the observation--action loop for one complete episode.
 * @tparam EnvironmentType Concrete Environment interface implementation.
 * @tparam PolicyType Concrete Policy implementation with matching observation
 * and action aliases.
 * @tparam EpisodeResultType Aggregate receiving reward and termination data.
 * @see simulation_execution_chapter
 */
template< typename EnvironmentType, typename PolicyType, typename EpisodeResultType = EpisodeResult<> >
class EpisodeRunner
{
  static_assert( std::is_same_v< typename EnvironmentType::Observation, typename PolicyType::Observation >,
                 "Policy and Environment Observation type differs" );
  static_assert( std::is_same_v< typename EnvironmentType::Action, typename PolicyType::Action >,
                 "Policy and Environment Action type differs" );

  public:
  using Environment = EnvironmentType; ///< Environment instance type.
  using Policy = PolicyType;            ///< Policy instance type.
  using Observation = typename Environment::Observation; ///< Shared observation type.
  using Action = typename Environment::Action;            ///< Shared action type.
  using Scenario = typename Environment::Scenario;        ///< Episode scenario type.
  using StepResult = typename Environment::StepResult;    ///< Per-transition result type.
  using TerminationReason = typename Environment::TerminationReason; ///< Ending reason type.
  using EpisodeResult = EpisodeResultType; ///< Aggregate returned to the caller.

  /**
   * @brief Binds the components used by one episode execution.
   * @param[in,out] environment Stateful environment used for the episode.
   * @param[in,out] policy Policy used to select every action.
   * @param[in] scenario Scenario forwarded to Environment::reset().
   * @pre The environment, policy, and scenario outlive this runner.
   */
  EpisodeRunner( Environment &environment, Policy &policy, const Scenario &scenario ) :
    environment_( environment ), policy_( policy ), scenario_( scenario )
  {}

  /** @brief Enables destruction through the runner interface. */
  virtual ~EpisodeRunner() = default;

  /**
   * @brief Runs one episode and returns its aggregate result.
   *
   * The method calls Environment::reset() once, then alternates Policy::act()
   * and Environment::step(). Rewards are accumulated in transition order. The
   * loop ends when StepResult::hasFinished() returns `true`; the final flags and
   * reason are copied to the returned result.
   *
   * @return Episode aggregate with total reward, step count, and final status.
   * @pre The environment eventually returns a finished StepResult.
   */
  virtual EpisodeResult run()
  {
    Observation observation = environment_.reset( scenario_ );

    EpisodeResult result;
    while ( true )
    {
      Action action = policy_.act( observation );
      StepResult step_result = environment_.step( action );

      result.total_reward += step_result.reward;
      result.steps++;

      observation = step_result.observation;
      if ( step_result.hasFinished() )
      {
        result.terminated = step_result.terminated;
        result.truncated = step_result.truncated;
        result.termination_reason = step_result.termination_reason;
        break;
      }
    }
    return result;
  }

  protected:
  /** Non-owning environment reference used for reset and step operations. */
  Environment &environment_;
  /** Non-owning policy reference used to select actions. */
  Policy &policy_;
  /** Non-owning scenario reference used to reset the environment. */
  const Scenario &scenario_;
};
/** @} */

#endif // !EPISODE_RUNNER_H
