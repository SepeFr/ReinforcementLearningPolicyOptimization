#ifndef STEP_RESULT_H
#define STEP_RESULT_H

/** @addtogroup simulation_api
 * @{ */


#include "ObservationBase.hpp"
#include "TerminationReason.hpp"

/**
 * @brief Result of one Environment::step() transition.
 * @tparam ObservationType Observation available after the transition.
 * @tparam TerminationReasonType Enum-like reason type containing a `None` value.
 */
template< typename ObservationType = ObservationBase, typename TerminationReasonType = TerminationReason >
class StepResult
{
  public:
  ObservationType observation; ///< Observation of the state reached by the transition.
  double reward = 0.0;         ///< Reward contributed by this transition.
  bool terminated = false;     ///< Whether an environment terminal state was reached.
  bool truncated = false;      ///< Whether an external limit ended the episode.
  TerminationReasonType termination_reason = TerminationReasonType::None; ///< Specific reason, or `None`.

  /**
   * @brief Tests whether an episode runner must stop after this transition.
   * @return `true` if either flag is set or termination_reason is not `None`.
   */
  virtual bool hasFinished() const
  {
    return terminated || truncated || termination_reason != TerminationReasonType::None;
  }
};

/** @} */

#endif // !STEP_RESULT_H
