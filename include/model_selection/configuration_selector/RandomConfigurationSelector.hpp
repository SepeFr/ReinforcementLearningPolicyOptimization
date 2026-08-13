#ifndef RANDOM_CONFIGURATION_SELECTOR_H
#define RANDOM_CONFIGURATION_SELECTOR_H

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
 * @brief Samples every configuration axis independently for each request.
 * @tparam Axes Non-empty sequence of configuration axis types.
 *
 * Discrete and categorical domains are sampled uniformly by stored index.
 * Continuous domains use `std::uniform_real_distribution` over their bounds.
 */
template< typename... Axes >
class RandomConfigurationSelector final : public ConfigurationSelector
{
  static_assert( sizeof...( Axes ) > 0, "RandomConfigurationSelector requires at least one configuration axis" );

  public:
  /**
   * @brief Creates a reproducible independent sampler.
   * @param[in,out] configuration Stable object whose axis fields are assigned before copying.
   * @param[in] search_space Axes pointing into `configuration`.
   * @param[in] random_seed Seed for the selector's `std::mt19937_64` generator.
   * @pre `configuration` remains alive and unmoved for this selector's lifetime.
   */
  RandomConfigurationSelector( ModelConfiguration &configuration, ConfigurationSearchSpace< Axes... > search_space,
                               std::uint64_t random_seed = 0 ) :
      configuration_( &configuration ), search_space_( std::move( search_space ) ), generator_( random_seed )
  {
  }

  /** @brief Reports that the sample stream requires external stopping. @return `true`. */
  bool requiresStoppingBudget() const noexcept override { return true; }
  /** @brief Reports availability of the unbounded random stream. @return `true`. */
  bool hasNext() const override { return true; }

  /**
   * @brief Independently samples a configuration for every batch position.
   * @param[in] maximum_batch_size Positive returned cardinality.
   * @return Exactly `maximum_batch_size` configuration copies.
   * @throws std::invalid_argument If `maximum_batch_size` is zero.
   */
  std::vector< ModelConfiguration > ask( std::size_t maximum_batch_size ) override
  {
    if ( maximum_batch_size == 0 )
    {
      throw std::invalid_argument( "RandomConfigurationSelector: maximum batch size must be greater than zero" );
    }

    std::vector< ModelConfiguration > configurations;
    configurations.reserve( maximum_batch_size );

    for ( std::size_t index = 0; index < maximum_batch_size; ++index )
    {
      sampleConfiguration();
      configurations.push_back( *configuration_ );
    }

    return configurations;
  }

  /** @brief Accepts and ignores feedback because independent sampling is non-adaptive. */
  void tell( const std::vector< ConfigurationFeedback > & ) override {}

  private:
  /**
   * @brief Samples one axis and assigns its bound field.
   * @tparam Axis Configuration axis type.
   * @param[in] axis Axis to sample.
   */
  template< typename Axis >
  void sample( const Axis &axis )
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

  /** @brief Samples all axes once in tuple order. */
  void sampleConfiguration()
  {
    std::apply( [this]( const auto &...axis ) { ( sample( axis ), ... ); }, search_space_.axes() );
  }

  ModelConfiguration *configuration_; ///< Non-owning pointer to the stable configuration template.
  ConfigurationSearchSpace< Axes... > search_space_; ///< Owned axes retaining pointers into `configuration_`.
  std::mt19937_64 generator_; ///< Persistent generator defining the reproducible sample stream.
};

/** @} */

#endif // !RANDOM_CONFIGURATION_SELECTOR_H
