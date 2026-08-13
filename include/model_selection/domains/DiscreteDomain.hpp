#ifndef DISCRETE_DOMAIN_H
#define DISCRETE_DOMAIN_H

/** @addtogroup model_selection_api
 * @{ */

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>
#include "DomainKind.hpp"

/**
 * @brief Defines an ordered finite sequence of configuration values.
 * @tparam T Copyable value type supporting equality comparison.
 *
 * Repeated values are retained and therefore occupy repeated grid positions
 * and receive repeated probability mass during index-based sampling.
 */
template< typename T >
class DiscreteDomain
{
  public:
  using ValueType = T; ///< Value stored in the configuration field.
  static constexpr DomainKind kind = DomainKind::Discrete; ///< Compile-time selector dispatch category.

  /**
   * @brief Creates a domain from an ordered non-empty sequence.
   * @param[in] values Values in grid-enumeration and index-sampling order.
   * @throws std::invalid_argument If `values` is empty.
   */
  explicit DiscreteDomain( std::vector< T > values ) : values_( std::move( values ) )
  {
    if ( values_.empty() )
    {
      throw std::invalid_argument( "DiscreteDomain: at least one value is required" );
    }
  }

  /** @brief Returns the ordered values. @return Reference valid for this domain's lifetime. */
  const std::vector< T > &values() const { return values_; }

  /**
   * @brief Tests whether the sequence contains a value.
   * @param[in] value Value compared by equality with stored entries.
   * @return `true` when at least one matching entry exists.
   */
  bool contains( const T &value ) const { return std::find( values_.begin(), values_.end(), value ) != values_.end(); }

  private:
  std::vector< T > values_; ///< Ordered, non-empty value sequence.
};

/** @} */

#endif // !DISCRETE_DOMAIN_H
