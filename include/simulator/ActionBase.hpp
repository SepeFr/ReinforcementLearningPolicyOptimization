#ifndef ACTION_BASE_H
#define ACTION_BASE_H

/**
 * @defgroup simulation_api Simulation
 * @brief Environment, policy, and episode-execution contracts.
 * @{
 */

#include <eigen3/Eigen/Core>
#include <utility>

/**
 * @brief Vector-backed action exchanged between a policy and an environment.
 *
 * The default conversion stores a copy of the supplied vector. Derived action
 * types may override vectorToAction() to decode a policy output into a richer
 * representation.
 *
 * @see ObservationBase
 * @see Policy
 * @see Environment
 */
class ActionBase
{
  public:
  /** @brief Constructs an action with an empty value vector. */
  ActionBase() = default;
  /**
   * @brief Constructs an action from a vector value.
   * @param[in] value Action components in environment-defined order.
   */
  explicit ActionBase( Eigen::VectorXd value ) : value_( std::move( value ) ) {}
  /** @brief Enables destruction through the base interface. */
  virtual ~ActionBase() = default;

  /**
   * @brief Replaces the action with components produced by a policy.
   * @param[in] vector Action components in environment-defined order. The
   * input is copied and need not outlive the call.
   * @post value() contains a copy of @p vector.
   */
  virtual void vectorToAction( Eigen::Ref< const Eigen::VectorXd > vector ) { value_ = vector; }

  /** @brief Returns the stored action components. @return Read-only reference valid for this object's lifetime. */
  const Eigen::VectorXd &value() const { return value_; }
  /** @brief Provides mutable access to the stored action components. @return Mutable reference valid for this object's lifetime. */
  Eigen::VectorXd &value() { return value_; }

  private:
  /** Action components in environment-defined order. */
  Eigen::VectorXd value_;
};

/**
 * @brief Typed convenience wrapper for vector-compatible actions.
 * @tparam VectorType Value accepted by Eigen::VectorXd construction.
 */
template< typename VectorType = Eigen::VectorXd >
class ActionVector : public ActionBase
{
  public:
  /** @brief Constructs an action with an empty value vector. */
  ActionVector() = default;
  /**
   * @brief Converts and stores a typed vector value.
   * @param[in] value Action components in environment-defined order.
   */
  explicit ActionVector( VectorType value ) : ActionBase( Eigen::VectorXd( std::move( value ) ) ) {}
};

/** @} */

#endif // !ACTION_BASE_H
