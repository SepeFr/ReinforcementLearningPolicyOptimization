#ifndef RANDOM_WALK_CONFIGURATION_SELECTOR_H
#define RANDOM_WALK_CONFIGURATION_SELECTOR_H

/** @addtogroup model_selection_api
 * @{ */

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "ConfigurationSearchSpace.hpp"
#include "ConfigurationSelector.hpp"
#include "DomainKind.hpp"

/**
 * @brief Produces a random walk by resampling one uniformly chosen axis per step.
 * @tparam Axes Non-empty sequence of configuration axis types.
 *
 * Each returned element follows one mutation of the previous configuration.
 * The first returned element therefore already differs by one sampled axis,
 * subject to the sampled value possibly equaling its previous value.
 */
template< typename... Axes >
class RandomWalkConfigurationSelector final : public ConfigurationSelector
{
  static_assert( sizeof...( Axes ) > 0,
                 "RandomWalkConfigurationSelector requires at least one configuration axis" );

  public:
  /**
   * @brief Creates a reproducible walk starting from the current configuration.
   * @param[in,out] configuration Stable state mutated as the walk advances.
   * @param[in] search_space Axes pointing into `configuration`.
   * @param[in] random_seed Seed for axis selection and replacement-value sampling.
   * @pre `configuration` remains alive and unmoved for this selector's lifetime.
   */
  RandomWalkConfigurationSelector( ModelConfiguration &configuration,
                                   ConfigurationSearchSpace< Axes... > search_space,
                                   std::uint64_t random_seed = 0 ) :
      configuration_( &configuration ), search_space_( std::move( search_space ) ), generator_( random_seed )
  {
  }

  /** @brief Reports that the walk requires external stopping. @return `true`. */
  bool requiresStoppingBudget() const noexcept override { return true; }
  /** @brief Reports availability of the unbounded walk. @return `true`. */
  bool hasNext() const override { return true; }

  /**
   * @brief Advances the walk once for every requested batch element.
   * @param[in] maximum_batch_size Positive returned cardinality.
   * @return Exactly `maximum_batch_size` sequential configuration states.
   * @throws std::invalid_argument If `maximum_batch_size` is zero.
   */
  std::vector< ModelConfiguration > ask( std::size_t maximum_batch_size ) override
  {
    if ( maximum_batch_size == 0 )
    {
      throw std::invalid_argument( "RandomWalkConfigurationSelector: maximum batch size must be greater than zero" );
    }

    std::vector< ModelConfiguration > configurations;
    configurations.reserve( maximum_batch_size );
    std::uniform_int_distribution< std::size_t > axis_distribution( 0, sizeof...( Axes ) - 1 );

    for ( std::size_t index = 0; index < maximum_batch_size; ++index )
    {
      mutateAxis( axis_distribution( generator_ ) );
      configurations.push_back( *configuration_ );
    }

    return configurations;
  }

  /** @brief Accepts and ignores feedback because walk transitions are non-adaptive. */
  void tell( const std::vector< ConfigurationFeedback > & ) override {}

  private:
  /**
   * @brief Resamples one axis value from its complete domain.
   * @tparam Axis Configuration axis type.
   * @param[in] axis Axis whose field is assigned.
   */
  template< typename Axis >
  void mutate( const Axis &axis )
  {
    using Domain = std::remove_cvref_t< decltype( axis.domain() ) >;

    const Domain &domain = axis.domain();

    if constexpr ( Domain::kind == DomainKind::Discrete )
    {
      std::uniform_int_distribution< std::size_t > distribution( 0, domain.values().size() - 1 );
      axis.setValue( domain.values().at( distribution( generator_ ) ) );
    }
    else if constexpr ( Domain::kind == DomainKind::Categorical )
    {
      std::uniform_int_distribution< std::size_t > distribution( 0, domain.choices().size() - 1 );
      axis.setValue( domain.choices().at( distribution( generator_ ) ).value );
    }
    else
    {
      std::uniform_real_distribution< typename Axis::ValueType > distribution( domain.lowerBound(),
                                                                               domain.upperBound() );
      axis.setValue( distribution( generator_ ) );
    }
  }

  /**
   * @brief Dispatches a runtime axis index to its tuple element.
   * @tparam Index Current compile-time tuple position.
   * @param[in] selected_axis Valid index in `[0, sizeof...(Axes))`.
   */
  template< std::size_t Index = 0 >
  void mutateAxis( std::size_t selected_axis )
  {
    if constexpr ( Index < sizeof...( Axes ) )
    {
      if ( Index == selected_axis )
      {
        mutate( std::get< Index >( search_space_.axes() ) );
        return;
      }

      mutateAxis< Index + 1 >( selected_axis );
    }
  }

  ModelConfiguration *configuration_; ///< Non-owning pointer to the persistent walk state.
  ConfigurationSearchSpace< Axes... > search_space_; ///< Owned axes retaining pointers into `configuration_`.
  std::mt19937_64 generator_; ///< Persistent generator for axes and replacement values.
};

/** @} */

#endif // !RANDOM_WALK_CONFIGURATION_SELECTOR_H
