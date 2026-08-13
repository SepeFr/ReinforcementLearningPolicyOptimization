#ifndef CONFIGURATION_SEARCH_SPACE_H
#define CONFIGURATION_SEARCH_SPACE_H

/** @addtogroup model_selection_api
 * @{ */

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include "ConfigurationAxis.hpp"

/**
 * @brief Defines the Cartesian product of configuration axes.
 * @tparam Axes ConfigurationAxis specializations stored in tuple order.
 *
 * Axis order defines genome order and grid nesting. Every axis must bind a
 * distinct field. The axes retain their non-owning field pointers.
 *
 * @see model_selection_chapter
 */
template< typename... Axes >
class ConfigurationSearchSpace
{
  public:
  /**
   * @brief Creates a search space and verifies field identity.
   * @param[in] axes Axes stored in the order supplied.
   * @throws std::invalid_argument If two axes point to the same field.
   */
  explicit ConfigurationSearchSpace( Axes... axes ) : axes_( std::move( axes )... ) { validateDuplicates(); }

  /** @brief Returns axes in search-space order. @return Const reference to the owned tuple. */
  const std::tuple< Axes... > &axes() const { return axes_; }

  private:
  /**
   * @brief Enforces the one-axis-per-field invariant.
   * @throws std::invalid_argument If two stored field addresses are equal.
   */
  void validateDuplicates() const
  {
    const std::array< const void *, sizeof...( Axes ) > parameters = std::apply(
      []( const auto &...axis ) { return std::array< const void *, sizeof...( Axes ) >{ axis.address()... }; }, axes_ );

    for ( std::size_t left = 0; left < parameters.size(); ++left )
    {
      for ( std::size_t right = left + 1; right < parameters.size(); ++right )
      {
        if ( parameters[left] == parameters[right] )
        {
          throw std::invalid_argument( "ConfigurationSearchSpace: duplicate configuration axis" );
        }
      }
    }
  }

  std::tuple< Axes... > axes_; ///< Axes in grid, genome, and sampling order.
};

// class template argument deduction guide
// this way we can write
// ConfigurationAxis<int> axis1;
// ConfigurationAxis<double> axis2;
// ConfigurationSearchSpace space(axis1, axis2);
//
// Deduction guide infers axis types from constructor arguments:
// ConfigurationSearchSpace<ConfigurationAxis< int >, ConfigurationAxis< double > > space( axis1, axis2 );
/** @brief Deduces all axis types from constructor arguments. */
template< typename... Axes >
ConfigurationSearchSpace( Axes... ) -> ConfigurationSearchSpace< Axes... >;

/** @} */

#endif // !CONFIGURATION_SEARCH_SPACE_H
