#ifndef GENETIC_CONFIGURATION_SELECTOR_H
#define GENETIC_CONFIGURATION_SELECTOR_H

/** @addtogroup model_selection_api
 * @{ */

#include <algorithm>
#include <cmath>
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
#include "GeneticAlgorithmEngine.hpp"
#include "GeneticConfigurationOperatorFactory.hpp"
#include "GeneticConfigurationSelectorConfiguration.hpp"

/**
 * @brief Adapts configuration fields to a generation-based genetic search.
 * @tparam Axes Non-empty axis sequence defining tuple genome positions.
 *
 * A complete generation may be emitted through several batches. Feedback is
 * retained in generation order and the GeneticAlgorithmEngine advances only
 * after every genome has received feedback. For successful evaluations, lower
 * selection scores map to higher fitness through
 * \f$F_i=S_{worst}-S_i+\varepsilon\f$. Failed evaluations receive zero fitness
 * and are excluded by the engine. A generation with no successful evaluation
 * exhausts the selector.
 *
 * @see GeneticAlgorithmEngine
 * @see genetic_search_chapter
 */
template< typename... Axes >
class GeneticConfigurationSelector final : public ConfigurationSelector
{
  static_assert( sizeof...( Axes ) > 0, "GeneticConfigurationSelector requires at least one configuration axis" );

  private:
  using Genome = std::tuple< typename Axes::ValueType... >; ///< Tuple of axis values in search-space order.
  using Crossover = CrossoverFunction< Genome >; ///< Type-erased configured crossover callable.
  using Mutation = MutationFunction< Genome >; ///< Type-erased configured mutation callable.

  public:
  /**
   * @brief Creates a genetic selector and samples generation zero.
   * @param[in,out] configuration Stable template whose axis fields are set before copying.
   * @param[in] search_space Axes pointing into `configuration` and defining genome order.
   * @param[in] selector_configuration Population, operator, seed, and fitness settings.
   * @throws std::invalid_argument If genetic settings, operator probabilities,
   *         or `fitness_epsilon` violate their documented ranges.
   * @pre `configuration` remains alive and unmoved for this selector's lifetime.
   */
  GeneticConfigurationSelector( ModelConfiguration &configuration, ConfigurationSearchSpace< Axes... > search_space,
                                GeneticConfigurationSelectorConfiguration selector_configuration ) :
      configuration_( &configuration ), search_space_( std::move( search_space ) ),
      selector_configuration_( std::move( selector_configuration ) ),
      initialization_generator_( selector_configuration_.algorithm.random_seed ),
      engine_( selector_configuration_.algorithm,
               sampleInitialPopulation( search_space_, selector_configuration_.algorithm.population_size,
                                        initialization_generator_ ),
               GeneticConfigurationOperatorFactory::createCrossover< Genome >(
                 selector_configuration_.crossover ),
               GeneticConfigurationOperatorFactory::createMutation< Axes... >( selector_configuration_.mutation,
                                                                                search_space_ ) )
  {
    if ( !std::isfinite( selector_configuration_.fitness_epsilon ) ||
         selector_configuration_.fitness_epsilon <= 0.0 )
    {
      throw std::invalid_argument( "GeneticConfigurationSelector: fitness epsilon must be finite and positive" );
    }
  }

  /** @brief Reports that continuing successful generations require external stopping. @return `true`. */
  bool requiresStoppingBudget() const noexcept override { return true; }
  /** @brief Reports whether the selector has a generation available. @return `false` after an all-failed generation. */
  bool hasNext() const override { return !exhausted_; }

  /**
   * @brief Requests the next slice of the current generation.
   *
   * A new complete generation is obtained from the engine when needed. Genome
   * values are assigned to the bound fields and the full model configuration is
   * copied for each returned element.
   *
   * @param[in] maximum_batch_size Positive upper limit for this generation slice.
   * @return Consecutive configurations in current-generation order.
   * @throws std::invalid_argument If `maximum_batch_size` is zero.
   * @throws std::logic_error If feedback for the preceding slice is pending or
   *         the caller requests a generation after exhaustion.
   * @pre `hasNext()` is `true`.
   * @post A non-empty result requires one matching `tell()` before another `ask()`.
   */
  std::vector< ModelConfiguration > ask( std::size_t maximum_batch_size ) override
  {
    if ( maximum_batch_size == 0 )
    {
      throw std::invalid_argument( "GeneticConfigurationSelector: maximum batch size must be greater than zero" );
    }
    if ( pending_batch_size_ != 0 )
    {
      throw std::logic_error( "GeneticConfigurationSelector: ask called while waiting for feedback" );
    }

    if ( current_generation_.empty() )
    {
      current_generation_ = engine_.askGeneration();
      generation_feedback_.clear();
      next_genome_index_ = 0;
    }

    const std::size_t remaining_genomes = current_generation_.size() - next_genome_index_;
    pending_batch_size_ = std::min( maximum_batch_size, remaining_genomes );

    std::vector< ModelConfiguration > configurations;
    configurations.reserve( pending_batch_size_ );
    for ( std::size_t offset = 0; offset < pending_batch_size_; ++offset )
    {
      configurations.push_back( makeConfiguration( current_generation_.at( next_genome_index_ + offset ) ) );
    }

    next_genome_index_ += pending_batch_size_;
    return configurations;
  }

  /**
   * @brief Appends feedback for the pending generation slice.
   *
   * Once feedback covers the complete generation, the method computes genetic
   * fitness and either advances the engine or marks the selector exhausted.
   *
   * @param[in] feedback Outcomes matching the pending configurations in order.
   * @throws std::logic_error If no batch is pending.
   * @throws std::invalid_argument If cardinality differs from the pending batch
   *         or a successful selection score is non-finite.
   * @pre Every successful entry has an engaged, finite `selection_score`.
   */
  void tell( const std::vector< ConfigurationFeedback > &feedback ) override
  {
    if ( pending_batch_size_ == 0 )
    {
      throw std::logic_error( "GeneticConfigurationSelector: tell called without a pending batch" );
    }
    if ( feedback.size() != pending_batch_size_ )
    {
      throw std::invalid_argument( "GeneticConfigurationSelector: feedback count does not match pending batch size" );
    }

    generation_feedback_.insert( generation_feedback_.end(), feedback.begin(), feedback.end() );
    pending_batch_size_ = 0;

    if ( generation_feedback_.size() == current_generation_.size() )
    {
      updateEngine();
    }
  }

  private:
  /**
   * @brief Samples one value using the axis domain's native distribution.
   * @tparam Axis Configuration axis type.
   * @param[in] axis Axis whose domain is sampled.
   * @param[in,out] generator Random stream.
   * @return A stored discrete/categorical value or continuous interval sample.
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

  /**
   * @brief Samples every genome position from its corresponding axis.
   * @tparam Indices Compile-time axis positions.
   * @param[in] axes Axis tuple in genome order.
   * @param[in,out] generator Random stream.
   * @return Independently sampled genome.
   */
  template< std::size_t... Indices >
  static Genome sampleGenome( const std::tuple< Axes... > &axes, std::mt19937_64 &generator,
                              std::index_sequence< Indices... > )
  {
    return Genome{ sampleValue( std::get< Indices >( axes ), generator )... };
  }

  /**
   * @brief Samples the requested number of initial genomes.
   * @param[in] search_space Domains defining genome positions.
   * @param[in] population_size Number of genomes to create.
   * @param[in,out] generator Random stream seeded by the selector configuration.
   * @return Initial population in sampling order.
   */
  static std::vector< Genome > sampleInitialPopulation( const ConfigurationSearchSpace< Axes... > &search_space,
                                                        std::size_t population_size, std::mt19937_64 &generator )
  {
    std::vector< Genome > population;
    population.reserve( population_size );

    for ( std::size_t individual = 0; individual < population_size; ++individual )
    {
      population.push_back(
        sampleGenome( search_space.axes(), generator, std::index_sequence_for< Axes... >{} ) );
    }

    return population;
  }

  /**
   * @brief Assigns every genome value to its bound configuration field.
   * @tparam Indices Compile-time genome and axis positions.
   * @param[in] genome Source values in axis order.
   */
  template< std::size_t... Indices >
  void applyGenome( const Genome &genome, std::index_sequence< Indices... > )
  {
    ( std::get< Indices >( search_space_.axes() ).setValue( std::get< Indices >( genome ) ), ... );
  }

  /**
   * @brief Materializes one genome as a complete model configuration.
   * @param[in] genome Axis values to apply.
   * @return Copy of `configuration_` after all axis assignments.
   */
  ModelConfiguration makeConfiguration( const Genome &genome )
  {
    applyGenome( genome, std::index_sequence_for< Axes... >{} );
    return *configuration_;
  }

  /**
   * @brief Converts complete generation feedback to fitness and closes the generation.
   *
   * The largest successful selection score becomes `worst_score`. Every
   * successful entry receives `worst_score - score + fitness_epsilon`; failed
   * entries receive zero failed fitness. An all-failed generation sets
   * `exhausted_` and clears selector-side pending state.
   *
   * @throws std::invalid_argument If a successful selection score is non-finite.
   */
  void updateEngine()
  {
    bool has_successful_evaluation = false;
    double worst_score = 0.0;

    for ( const ConfigurationFeedback &feedback : generation_feedback_ )
    {
      if ( feedback.status != ConfigurationEvaluationStatus::Succeeded )
      {
        continue;
      }
      const double selection_score = *feedback.evaluation.selection_score;
      if ( !std::isfinite( selection_score ) )
      {
        throw std::invalid_argument( "GeneticConfigurationSelector: successful selection score must be finite" );
      }

      if ( !has_successful_evaluation || selection_score > worst_score )
      {
        worst_score = selection_score;
        has_successful_evaluation = true;
      }
    }

    if ( !has_successful_evaluation )
    {
      exhausted_ = true;
      pending_batch_size_ = 0;
      current_generation_.clear();
      generation_feedback_.clear();
      next_genome_index_ = 0;
      return;
    }

    std::vector< GeneticFitness > fitness;
    fitness.reserve( generation_feedback_.size() );
    for ( const ConfigurationFeedback &feedback : generation_feedback_ )
    {
      if ( feedback.status == ConfigurationEvaluationStatus::Succeeded )
      {
        fitness.push_back( GeneticFitness{ worst_score - *feedback.evaluation.selection_score +
                                            selector_configuration_.fitness_epsilon,
                                          GeneticEvaluationStatus::Succeeded } );
      }
      else
      {
        fitness.push_back( GeneticFitness{ 0.0, GeneticEvaluationStatus::Failed } );
      }
    }

    engine_.tellGeneration( fitness );
    current_generation_.clear();
    generation_feedback_.clear();
    next_genome_index_ = 0;
  }

  ModelConfiguration *configuration_; ///< Non-owning pointer to the stable configuration template.
  ConfigurationSearchSpace< Axes... > search_space_; ///< Owned axes retaining pointers into `configuration_`.
  GeneticConfigurationSelectorConfiguration selector_configuration_; ///< Fixed evolution and score-mapping settings.
  std::mt19937_64 initialization_generator_; ///< Stream used only to sample generation zero.
  GeneticAlgorithmEngine< Genome, Crossover, Mutation > engine_; ///< Owner of population evolution after initialization.

  std::vector< Genome > current_generation_; ///< Engine generation retained while batches are emitted.
  std::vector< ConfigurationFeedback > generation_feedback_; ///< Ordered feedback accumulated across generation slices.
  std::size_t next_genome_index_ = 0; ///< First genome not yet returned from the current generation.
  std::size_t pending_batch_size_ = 0; ///< Cardinality expected by the next `tell()`.
  bool exhausted_ = false; ///< Set when a complete generation has no successful evaluation.
};

/** @} */

#endif // !GENETIC_CONFIGURATION_SELECTOR_H
