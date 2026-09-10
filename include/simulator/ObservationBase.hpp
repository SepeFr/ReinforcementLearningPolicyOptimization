#ifndef OBSERVATION_BASE_H
#define OBSERVATION_BASE_H

/** @addtogroup simulation_api
 * @{ */

#include <eigen3/Eigen/Core>
#include <utility>

/**
 * @brief Vector-backed observation exposed by an environment.
 *
 * The default conversion returns a copy. Derived observations may override
 * observationToVector() to encode structured state for a vector policy.
 */
class ObservationBase
{
  public:
  /** @brief Constructs an observation with an empty value vector. */
  ObservationBase() = default;
  /**
   * @brief Constructs an observation from vector components.
   * @param[in] value Components in policy-input order.
   */
  explicit ObservationBase( Eigen::VectorXd value ) : value_( std::move( value ) ) {}
  /** @brief Enables destruction through the base interface. */
  virtual ~ObservationBase() = default;

  /**
   * @brief Encodes this observation for a vector policy.
   * @return Copy of the components in policy-input order.
   */
  virtual Eigen::VectorXd observationToVector() const { return value_; }

  /** @brief Returns the stored observation components. @return Read-only reference valid for this object's lifetime. */
  const Eigen::VectorXd &value() const { return value_; }
  /** @brief Provides mutable access to the stored observation components. @return Mutable reference valid for this
   * object's lifetime. */
  Eigen::VectorXd &value() { return value_; }

  private:
  /** Observation components in policy-input order. */
  Eigen::VectorXd value_;
};

/**
 * @brief Typed convenience wrapper for vector-compatible observations.
 * @tparam VectorType Value accepted by Eigen::VectorXd construction.
 */
template< typename VectorType = Eigen::VectorXd >
class ObservationVector : public ObservationBase
{
  public:
  /** @brief Constructs an observation with an empty value vector. */
  ObservationVector() = default;
  /**
   * @brief Converts and stores a typed observation vector.
   * @param[in] value Components in policy-input order.
   */
  explicit ObservationVector( VectorType value ) : ObservationBase( Eigen::VectorXd( std::move( value ) ) ) {}
};

/** @} */

#endif // !OBSERVATION_BASE_H
