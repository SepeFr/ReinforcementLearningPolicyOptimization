#ifndef GRID_CONFIGURATION_SELECTOR_H
#define GRID_CONFIGURATION_SELECTOR_H

/** @addtogroup model_selection_api
 * @{ */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "ConfigurationSearchSpace.hpp"
#include "ConfigurationSelector.hpp"
#include "DomainKind.hpp"

/**
 * @brief Enumerates the Cartesian product of a configuration search space.
 * @tparam Axes Axis types in nesting order.
 *
 * The last axis changes fastest. Discrete and categorical values retain domain
 * order. A continuous interval uses `ceil((upper-lower)/step)+1` points; the
 * final point is clamped to the inclusive upper bound.
 *
 * @see model_selection_chapter
 */
template< typename... Axes >
class GridConfigurationSelector final : public ConfigurationSelector
{
  public:
  /**
   * @brief Creates a finite grid selector over fields of `configuration`.
   * @param[in,out] configuration Object whose axis fields are assigned before each returned copy.
   * @param[in] search_space Axes that point into `configuration` and define enumeration order.
   * @param[in] continuous_step_size Positive spacing shared by all continuous axes.
   * @throws std::invalid_argument If `continuous_step_size` is non-finite or not positive.
   * @throws std::overflow_error If a continuous range or cardinality cannot be represented.
   * @pre `configuration` remains alive and unmoved for this selector's lifetime.
   */
  GridConfigurationSelector( ModelConfiguration &configuration, ConfigurationSearchSpace< Axes... > search_space,
                             double continuous_step_size ) :
      configuration_( &configuration ), search_space_( std::move( search_space ) ),
      continuous_step_size_( continuous_step_size )
  {
    if ( !std::isfinite( continuous_step_size_ ) || continuous_step_size_ <= 0.0 )
    {
      throw std::invalid_argument(
        "GridConfigurationSelector: continuous step size must be finite and greater than zero" );
    }

    cardinalities_ = std::apply( [this]( const auto &...axis )
                                 { return std::vector< std::size_t >{ cardinality( axis )... }; },
                                 search_space_.axes() );
    indices_.assign( cardinalities_.size(), 0 );
  }

  /** @brief Reports that grid exhaustion bounds the search. @return `false`. */
  bool requiresStoppingBudget() const noexcept override { return false; }
  /** @brief Reports whether an unvisited Cartesian-product point remains. @return Remaining-grid status. */
  bool hasNext() const override { return !indices_.empty() && indices_.front() < cardinalities_.front(); }

  /**
   * @brief Returns up to the requested number of consecutive grid points.
   * @param[in] maximum_batch_size Positive batch limit.
   * @return Configuration copies in mixed-radix order, possibly empty after exhaustion.
   * @throws std::invalid_argument If `maximum_batch_size` is zero.
   * @post The bound configuration contains the last generated point when the result is non-empty.
   */
  std::vector< ModelConfiguration > ask( std::size_t maximum_batch_size ) override
  {
    if ( maximum_batch_size == 0 )
    {
      throw std::invalid_argument( "GridConfigurationSelector: maximum batch size must be greater than zero" );
    }

    std::vector< ModelConfiguration > configurations;
    configurations.reserve( maximum_batch_size );

    while ( configurations.size() < maximum_batch_size && hasNext() )
    {
      applyCurrentIndices( std::index_sequence_for< Axes... >{} );
      configurations.push_back( *configuration_ );
      incrementIndices();
    }

    return configurations;
  }

  /** @brief Accepts and ignores feedback because grid enumeration is non-adaptive. */
  void tell( const std::vector< ConfigurationFeedback > & ) override {}

  private:
  /**
   * @brief Computes one axis cardinality under grid sampling rules.
   * @tparam Axis Configuration axis type.
   * @param[in] axis Axis whose domain is inspected.
   * @return Stored-value count or continuous grid-point count.
   * @throws std::overflow_error If a continuous range or cardinality cannot be represented.
   */
  template< typename Axis >
  std::size_t cardinality( const Axis &axis ) const
  {
    using Domain = std::remove_cvref_t< decltype( axis.domain() ) >;

    if constexpr ( Domain::kind == DomainKind::Discrete )
    {
      return axis.domain().values().size();
    }
    else if constexpr ( Domain::kind == DomainKind::Categorical )
    {
      return axis.domain().choices().size();
    }
    else
    {
      if ( axis.domain().lowerBound() == axis.domain().upperBound() )
      {
        return 1;
      }

      const long double range = static_cast< long double >( axis.domain().upperBound() ) -
        static_cast< long double >( axis.domain().lowerBound() );
      const long double step = static_cast< long double >( continuous_step_size_ );
      if ( !std::isfinite( range ) || range < 0.0L )
      {
        throw std::overflow_error( "GridConfigurationSelector: continuous axis range is not representable" );
      }

      const long double interval_ratio = range / step;
      if ( !std::isfinite( interval_ratio ) || interval_ratio < 0.0L )
      {
        throw std::overflow_error( "GridConfigurationSelector: continuous axis cardinality is not representable" );
      }

      const long double interval_count = std::max( 1.0L, std::ceil( interval_ratio ) );
      const long double maximum_interval_count =
        static_cast< long double >( std::numeric_limits< std::size_t >::max() );
      if ( !std::isfinite( interval_count ) || interval_count < 0.0L ||
           interval_count >= maximum_interval_count )
      {
        throw std::overflow_error( "GridConfigurationSelector: continuous axis cardinality is not representable" );
      }

      const std::size_t represented_interval_count = static_cast< std::size_t >( interval_count );
      if ( represented_interval_count == std::numeric_limits< std::size_t >::max() )
      {
        throw std::overflow_error( "GridConfigurationSelector: continuous axis cardinality is not representable" );
      }

      // Add one grid index; applyValue clamps it so the inclusive upper bound is always represented.
      return represented_interval_count + 1;
    }
  }
  /**
   * @brief Assigns the value represented by one axis index.
   * @tparam Axis Configuration axis type.
   * @param[in] axis Axis whose pointed field is updated.
   * @param[in] index Valid index below the axis cardinality.
   */
  template< typename Axis >
  void applyValue( const Axis &axis, std::size_t index )
  {
    using Domain = std::remove_cvref_t< decltype( axis.domain() ) >;

    if constexpr ( Domain::kind == DomainKind::Discrete )
    {
      axis.setValue( axis.domain().values()[index] );
    }
    else if constexpr ( Domain::kind == DomainKind::Categorical )
    {
      axis.setValue( axis.domain().choices()[index].value );
    }
    else
    {
      const auto unclamped_value =
        axis.domain().lowerBound() + static_cast< typename Axis::ValueType >( index * continuous_step_size_ );
      axis.setValue( std::min( unclamped_value, axis.domain().upperBound() ) );
    }
  }

  /**
   * @brief Applies the complete mixed-radix index tuple to bound fields.
   * @tparam Indices Compile-time axis positions.
   */
  template< std::size_t... Indices >
  void applyCurrentIndices( std::index_sequence< Indices... > )
  {
    ( applyValue( std::get< Indices >( search_space_.axes() ), indices_[Indices] ), ... );
  }

  /** @brief Advances the mixed-radix counter with the last axis changing fastest. */
  void incrementIndices()
  {
    for ( std::size_t position = indices_.size(); position > 0; --position )
    {
      const std::size_t index = position - 1;
      ++indices_[index];

      if ( indices_[index] < cardinalities_[index] )
      {
        return;
      }

      if ( index == 0 )
      {
        return;
      }

      indices_[index] = 0;
    }
  }

  ModelConfiguration *configuration_; ///< Non-owning pointer to the stable configuration template.
  ConfigurationSearchSpace< Axes... > search_space_; ///< Owned axes retaining pointers into `configuration_`.
  double continuous_step_size_; ///< Shared positive spacing for continuous axes.
  std::vector< std::size_t > cardinalities_; ///< Available grid values per axis, in tuple order.
  std::vector< std::size_t > indices_; ///< Current mixed-radix position; first-axis overflow marks exhaustion.
};

/** @} */

#endif // !GRID_CONFIGURATION_SELECTOR_H
