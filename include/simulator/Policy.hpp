#ifndef POLICY_H
#define POLICY_H

/** @addtogroup simulation_api
 * @{ */

#include "ActionBase.hpp"
#include "ObservationBase.hpp"

/**
 * @brief Interface that maps environment observations to actions.
 * @tparam ObservationType Observation accepted by act().
 * @tparam ActionType Action returned to the environment.
 */
template< typename ObservationType = ObservationBase, typename ActionType = ActionBase >
class Policy
{
  public:
  using Observation = ObservationType; ///< Observation exchanged with the environment.
  using Action = ActionType;           ///< Action exchanged with the environment.

  /** @brief Enables destruction through the policy interface. */
  virtual ~Policy() = default;

  /**
   * @brief Selects an action for the current observation.
   * @param[in] observation Current environment observation.
   * @return Action to pass to Environment::step().
   */
  virtual Action act( const Observation &observation ) = 0;
};

/** @} */

#endif // !POLICY_H
