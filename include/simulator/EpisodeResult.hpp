#ifndef EPISODE_RESULT_H
#define EPISODE_RESULT_H

/** @addtogroup simulation_api
 * @{ */


#include <cstddef>
#include "TerminationReason.hpp"

/**
 * @brief Aggregate produced after an episode finishes.
 * @tparam TerminationReasonType Enum-like reason type containing a `None` value.
 */
template< typename TerminationReasonType = TerminationReason >
class EpisodeResult
{
  public:
  double total_reward = 0.0; ///< Sum of transition rewards in execution order.
  std::size_t steps = 0;     ///< Number of calls made to Environment::step().

  bool terminated = false; ///< Final transition's terminal-state flag.
  bool truncated = false;  ///< Final transition's external-limit flag.

  TerminationReasonType termination_reason = TerminationReasonType::None; ///< Final transition's reason.

  /**
   * @brief Tests whether this aggregate represents a finished episode.
   * @return `true` if either flag is set or termination_reason is not `None`.
   */
  virtual bool hasFinished() const
  {
    return terminated || truncated || termination_reason != TerminationReasonType::None;
  }
};

/** @} */

#endif // !EPISODE_RESULT_H
