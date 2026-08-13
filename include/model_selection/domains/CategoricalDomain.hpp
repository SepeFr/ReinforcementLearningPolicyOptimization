#ifndef CATEGORICAL_DOMAIN_H
#define CATEGORICAL_DOMAIN_H

/** @addtogroup model_selection_api
 * @{ */

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "DomainKind.hpp"

/**
 * @brief Associates a stable display identifier with a categorical value.
 * @tparam T Stored value type; equality comparison is required.
 */
template< typename T >
struct NamedChoice
{
  std::string identifier; ///< Identifier required to be unique within one domain.
  T value; ///< Configuration value required to be unique within one domain.
};

/**
 * @brief Defines a finite set of uniquely named configuration values.
 * @tparam T Copyable value type supporting equality comparison.
 * @see ConfigurationAxis
 */
template< typename T >
class CategoricalDomain
{
  public:
  using ValueType = T; ///< Value stored in the configuration field.
  static constexpr DomainKind kind = DomainKind::Categorical; ///< Compile-time selector dispatch category.

  /**
   * @brief Creates a domain from an ordered sequence of named choices.
   * @param[in] choices Choices in grid-enumeration and index-sampling order.
   * @throws std::invalid_argument If the sequence is empty or contains duplicate identifiers or values.
   */
  explicit CategoricalDomain( std::vector< NamedChoice< T > > choices ) : choices_( std::move( choices ) )
  {
    if ( choices_.empty() )
    {
      throw std::invalid_argument( "CategoricalDomain: at least one choice is required" );
    }

    for ( std::size_t left = 0; left < choices_.size(); ++left )
    {
      for ( std::size_t right = left + 1; right < choices_.size(); ++right )
      {
        if ( choices_[left].identifier == choices_[right].identifier || choices_[left].value == choices_[right].value )
        {
          throw std::invalid_argument( "CategoricalDomain: choices must have unique identifiers and values" );
        }
      }
    }
  }

  /** @brief Returns the ordered choices. @return Reference valid for this domain's lifetime. */
  const std::vector< NamedChoice< T > > &choices() const { return choices_; }

  /**
   * @brief Tests whether a value is one of the stored choices.
   * @param[in] value Value compared by equality with each choice.
   * @return `true` when a matching choice exists.
   */
  bool contains( const T &value ) const
  {
    for ( const NamedChoice< T > &choice : choices_ )
    {
      if ( choice.value == value )
      {
        return true;
      }
    }
    return false;
  }

  private:
  std::vector< NamedChoice< T > > choices_; ///< Ordered, non-empty sequence with unique identifiers and values.
};

/** @} */

#endif // !CATEGORICAL_DOMAIN_H
