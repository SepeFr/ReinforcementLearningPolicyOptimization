#ifndef TERMINATION_REASON_H
#define TERMINATION_REASON_H

/** @addtogroup simulation_api
 * @{ */

/** @file TerminationReason.hpp @brief Generic episode termination categories. */

/** @brief Generic reason associated with the end of an environment episode. */
enum class TerminationReason
{
  None,             ///< No termination reason has been reported.
  Success,          ///< The environment reached its success condition.
  Failure,          ///< The environment reached a failure condition.
  TimeLimitReached, ///< The environment exhausted its time or step limit.
  InvalidState,     ///< The environment detected an invalid simulation state.
};

/** @} */

#endif // !TERMINATION_REASON_H
