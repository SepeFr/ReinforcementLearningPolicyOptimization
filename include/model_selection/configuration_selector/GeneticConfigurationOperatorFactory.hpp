#ifndef GENETIC_CONFIGURATION_OPERATOR_FACTORY_H
#define GENETIC_CONFIGURATION_OPERATOR_FACTORY_H

/** @addtogroup model_selection_api
 * @{ */

/** @file GeneticConfigurationOperatorFactory.hpp @brief Tuple-genome crossover and mutation callable types and factory. */

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include "ConfigurationSearchSpace.hpp"
#include "DomainKind.hpp"
#include "GeneticConfigurationSelectorConfiguration.hpp"

/**
 * @brief Type-erased crossover callable for a configuration genome.
 * @tparam Genome Tuple-like genome type.
 */
template< typename Genome >
using CrossoverFunction =
  std::function< Genome( const Genome &, const Genome &, std::mt19937_64 & ) >;

/**
 * @brief Type-erased in-place mutation callable for a configuration genome.
 * @tparam Genome Tuple-like genome type.
 */
template< typename Genome >
using MutationFunction = std::function< void( Genome &, std::mt19937_64 & ) >;

/**
 * @brief Builds genetic operators whose tuple positions match search-space axes.
 *
 * @see GeneticConfigurationSelector
 * @see genetic_search_chapter
 */
class GeneticConfigurationOperatorFactory
{
  public:
  /** @brief Prevents construction of this stateless factory. */
  GeneticConfigurationOperatorFactory() = delete;

  /**
   * @brief Creates an independent uniform crossover callable.
   * @tparam Genome Tuple-like genome accepted by `std::get` and `std::tuple_size`.
   * @param[in] configuration Active crossover alternative and its probability.
   * @return Callable that selects every gene independently from either parent.
   * @throws std::invalid_argument If the first-parent probability is non-finite or outside `[0,1]`.
   */
  template< typename Genome >
  static CrossoverFunction< Genome >
  createCrossover( const ConfigurationCrossoverConfiguration &configuration )
  {
    return std::visit(
      []( const auto &selected_configuration ) -> CrossoverFunction< Genome >
      {
        if ( !std::isfinite( selected_configuration.first_parent_gene_probability ) ||
             selected_configuration.first_parent_gene_probability < 0.0 ||
             selected_configuration.first_parent_gene_probability > 1.0 )
        {
          throw std::invalid_argument(
            "GeneticConfigurationOperatorFactory: parent gene probability must be finite and between zero and one" );
        }

        return [selected_configuration]( const Genome &first_parent, const Genome &second_parent,
                                         std::mt19937_64 &generator )
        {
          return crossover( first_parent, second_parent,
                            selected_configuration.first_parent_gene_probability, generator,
                            std::make_index_sequence< std::tuple_size_v< Genome > >{} );
        };
      },
      configuration );
  }

  /**
   * @brief Creates a per-axis random-reset mutation callable.
   * @tparam Axes Search-space axis types defining genome positions and value types.
   * @param[in] configuration Active mutation alternative and its probability.
   * @param[in] search_space Domains copied into the returned callable in axis order.
   * @return Callable that independently resamples selected genes from their domains.
   * @throws std::invalid_argument If the per-axis probability is non-finite or outside `[0,1]`.
   */
  template< typename... Axes >
  static MutationFunction< std::tuple< typename Axes::ValueType... > >
  createMutation( const ConfigurationMutationConfiguration &configuration,
                  const ConfigurationSearchSpace< Axes... > &search_space )
  {
    using Genome = std::tuple< typename Axes::ValueType... >;

    return std::visit(
      [axes = search_space.axes()]( const auto &selected_configuration ) -> MutationFunction< Genome >
      {
        if ( !std::isfinite( selected_configuration.per_axis_probability ) ||
             selected_configuration.per_axis_probability < 0.0 ||
             selected_configuration.per_axis_probability > 1.0 )
        {
          throw std::invalid_argument(
            "GeneticConfigurationOperatorFactory: mutation probability must be finite and between zero and one" );
        }

        return [axes, selected_configuration]( Genome &genome, std::mt19937_64 &generator )
        {
          mutate( genome, axes, selected_configuration.per_axis_probability, generator,
                  std::index_sequence_for< Axes... >{} );
        };
      },
      configuration );
  }

  private:
  /**
   * @brief Implements per-gene uniform crossover over a compile-time index pack.
   * @tparam Genome Tuple-like genome type.
   * @tparam Indices Genome positions.
   * @param[in] first_parent First source genome.
   * @param[in] second_parent Second source genome and initial child value.
   * @param[in] first_parent_gene_probability Probability of selecting each first-parent gene.
   * @param[in,out] generator Random stream used for Bernoulli draws.
   * @return Child genome assembled independently at every position.
   */
  template< typename Genome, std::size_t... Indices >
  static Genome crossover( const Genome &first_parent, const Genome &second_parent,
                           double first_parent_gene_probability, std::mt19937_64 &generator,
                           std::index_sequence< Indices... > )
  {
    std::bernoulli_distribution select_first_parent( first_parent_gene_probability );
    Genome child = second_parent;

    ( selectGene< Indices >( child, first_parent, second_parent, select_first_parent, generator ), ... );
    return child;
  }

  /**
   * @brief Selects one gene from either parent and writes it into the child.
   * @tparam Index Genome position to assign.
   * @tparam Genome Tuple-like genome type.
   * @param[in,out] child Child receiving the selected gene.
   * @param[in] first_parent First source genome.
   * @param[in] second_parent Second source genome.
   * @param[in,out] select_first_parent Bernoulli distribution for the configured probability.
   * @param[in,out] generator Random stream used for selection.
   */
  template< std::size_t Index, typename Genome >
  static void selectGene( Genome &child, const Genome &first_parent, const Genome &second_parent,
                          std::bernoulli_distribution &select_first_parent, std::mt19937_64 &generator )
  {
    std::get< Index >( child ) = select_first_parent( generator ) ? std::get< Index >( first_parent )
                                                                  : std::get< Index >( second_parent );
  }

  /**
   * @brief Applies independent mutation decisions to all genome positions.
   * @tparam Genome Tuple-like genome type.
   * @tparam AxisTuple Tuple containing matching axis domains.
   * @tparam Indices Genome and axis positions.
   * @param[in,out] genome Genome mutated in place.
   * @param[in] axes Axis domains in genome order.
   * @param[in] per_axis_probability Independent mutation probability.
   * @param[in,out] generator Random stream for decisions and replacement values.
   */
  template< typename Genome, typename AxisTuple, std::size_t... Indices >
  static void mutate( Genome &genome, const AxisTuple &axes, double per_axis_probability,
                      std::mt19937_64 &generator, std::index_sequence< Indices... > )
  {
    std::bernoulli_distribution should_mutate( per_axis_probability );
    ( mutateGene< Indices >( genome, axes, should_mutate, generator ), ... );
  }

  /**
   * @brief Optionally replaces one gene with a domain sample.
   * @tparam Index Genome and axis position.
   * @tparam Genome Tuple-like genome type.
   * @tparam AxisTuple Tuple containing matching axis domains.
   * @param[in,out] genome Genome receiving a replacement when selected.
   * @param[in] axes Axis domains in genome order.
   * @param[in,out] should_mutate Bernoulli distribution for the configured probability.
   * @param[in,out] generator Random stream for the decision and possible sample.
   */
  template< std::size_t Index, typename Genome, typename AxisTuple >
  static void mutateGene( Genome &genome, const AxisTuple &axes, std::bernoulli_distribution &should_mutate,
                          std::mt19937_64 &generator )
  {
    if ( should_mutate( generator ) )
    {
      std::get< Index >( genome ) = sampleValue( std::get< Index >( axes ), generator );
    }
  }

  /**
   * @brief Draws one value according to an axis domain kind.
   * @tparam Axis Configuration axis type.
   * @param[in] axis Axis whose discrete indices, choices, or interval are sampled.
   * @param[in,out] generator Random stream.
   * @return Sampled value of `Axis::ValueType`.
   */
  template< typename Axis >
  static typename Axis::ValueType sampleValue( const Axis &axis, std::mt19937_64 &generator )
  {
    using Domain = std::remove_cvref_t< decltype( axis.domain() ) >;
    const Domain &domain = axis.domain();

    if constexpr ( Domain::kind == DomainKind::Discrete )
    {
      std::uniform_int_distribution< std::size_t > distribution( 0, domain.values().size() - 1 );
      return domain.values().at( distribution( generator ) );
    }
    else if constexpr ( Domain::kind == DomainKind::Categorical )
    {
      std::uniform_int_distribution< std::size_t > distribution( 0, domain.choices().size() - 1 );
      return domain.choices().at( distribution( generator ) ).value;
    }
    else
    {
      std::uniform_real_distribution< typename Axis::ValueType > distribution( domain.lowerBound(),
                                                                               domain.upperBound() );
      return distribution( generator );
    }
  }
};

/** @} */

#endif // !GENETIC_CONFIGURATION_OPERATOR_FACTORY_H
