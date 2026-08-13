#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

/** @addtogroup simulation_api
 * @{ */

#include "ActionBase.hpp"
#include "ObservationBase.hpp"
#include "StepResult.hpp"
#include "TerminationReason.hpp"

/**
 * @brief Stateful simulation interface driven by scenarios and policy actions.
 *
 * reset() starts a new episode and step() advances the same episode. A caller
 * must keep the environment alive and avoid interleaving episodes on one
 * instance unless the concrete implementation explicitly supports it.
 *
 * @tparam ScenarioType Input that configures one episode.
 * @tparam ObservationType State representation returned to the policy.
 * @tparam ActionType Control representation accepted from the policy.
 * @tparam TerminationReasonType Reason carried by each transition result.
 * @see EpisodeRunner
 */
template< typename ScenarioType, typename ObservationType = ObservationBase, typename ActionType = ActionBase,
          typename TerminationReasonType = TerminationReason >
class Environment
{
  public:
  using Observation = ObservationType; ///< Observation returned after reset and each step.
  using Scenario = ScenarioType;       ///< Scenario used to initialize an episode.
  using Action = ActionType;           ///< Action accepted for a transition.
  using TerminationReason = TerminationReasonType; ///< Episode-ending reason type.
  using StepResult = ::StepResult< Observation, TerminationReason >; ///< Transition result type.

  /** @brief Enables destruction through the environment interface. */
  virtual ~Environment() = default;

  /**
   * @brief Resets internal episode state for a scenario.
   * @param[in] scenario Scenario selected by the caller. The interface does
   * not retain ownership of it.
   * @return Initial observation of the new episode.
   * @post A subsequent step() advances the newly reset episode.
   */
  virtual Observation reset( const Scenario &scenario ) = 0;

  /**
   * @brief Advances the current episode by one action.
   * @param[in] action Action selected from the current observation.
   * @return Next observation, reward, and episode-ending state.
   * @pre reset() has established an episode for this instance.
   */
  virtual StepResult step( const Action &action ) = 0;
};

/** @} */

#endif // !ENVIRONMENT_H
