#ifndef CONTINUOUS_DOMAIN_H
#define CONTINUOUS_DOMAIN_H

/** @addtogroup model_selection_api
 * @{ */

#include <cmath>
#include <concepts>
#include <stdexcept>
#include "DomainKind.hpp"

/**
 * @brief Defines a finite closed interval for a floating-point configuration field.
 * @tparam T Floating-point value type.
 */
template< std::floating_point T >
class ContinuousDomain
{
  public:
  using ValueType = T; ///< Value stored in the configuration field.
  static constexpr DomainKind kind = DomainKind::Continuous; ///< Compile-time selector dispatch category.

  /**
   * @brief Creates the closed interval `[lower_bound, upper_bound]`.
   * @param[in] lower_bound Inclusive finite lower endpoint.
   * @param[in] upper_bound Inclusive finite upper endpoint.
   * @throws std::invalid_argument If an endpoint is non-finite or the bounds are reversed.
   */
  ContinuousDomain( T lower_bound, T upper_bound ) : lower_bound_( lower_bound ), upper_bound_( upper_bound )
  {
    if ( !std::isfinite( lower_bound_ ) || !std::isfinite( upper_bound_ ) || lower_bound_ > upper_bound_ )
    {
      throw std::invalid_argument( "ContinuousDomain: bounds must be finite and ordered" );
    }
  }

  /** @brief Returns the inclusive lower endpoint. @return Finite lower bound. */
  T lowerBound() const { return lower_bound_; }
  /** @brief Returns the inclusive upper endpoint. @return Finite upper bound. */
  T upperBound() const { return upper_bound_; }

  /**
   * @brief Tests membership in the closed interval.
   * @param[in] value Value to test.
   * @return `true` exactly when `lowerBound() <= value && value <= upperBound()`.
   */
  bool contains( T value ) const { return lower_bound_ <= value && value <= upper_bound_; }

  private:
  T lower_bound_; ///< Inclusive finite lower endpoint.
  T upper_bound_; ///< Inclusive finite upper endpoint.
};

/** @} */

#endif // !CONTINUOUS_DOMAIN_H
