#ifndef PARAMETRIZED_POLICY_H
#define PARAMETRIZED_POLICY_H

/** @addtogroup neural_network_api
 * @{ */

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <span>
#include "ActionBase.hpp"
#include "ObservationBase.hpp"
#include "Policy.hpp"

/**
 * @brief Policy interface whose behavior is controlled by a flat real vector.
 * @tparam ObservationType Observation accepted by Policy::act().
 * @tparam ActionType Action returned by Policy::act().
 */
template< typename ObservationType = ObservationBase, typename ActionType = ActionBase >
class ParametrizedPolicy : public Policy< ObservationType, ActionType >
{
  public:
  using ParameterVector = Eigen::VectorXd; ///< Owning representation returned by parameters().

  /** @brief Returns the required parameter-vector cardinality. @return Number of scalar parameters. */
  virtual std::size_t parameterCount() const = 0;

  /**
   * @brief Applies a contiguous parameter sequence.
   * @param[in] parameters Exactly parameterCount() values; the implementation copies them.
   */
  virtual void setParameters( std::span< const double > parameters ) = 0;
  /**
   * @brief Applies an Eigen parameter vector.
   * @param[in] parameters Exactly parameterCount() values; the implementation copies them.
   */
  virtual void setParameters( Eigen::Ref< const ParameterVector > parameters ) = 0;

  /** @brief Exports the current parameter values. @return Owning vector in the implementation's canonical order. */
  virtual ParameterVector parameters() const = 0;
};

/** @} */

#endif // !PARAMETRIZED_POLICY_H
