#ifndef GENETIC_ALGORITHM_ENGINE_H
#define GENETIC_ALGORITHM_ENGINE_H

/** @addtogroup genetic_algorithm_api
 * @{ */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "GeneticAlgorithmConfiguration.hpp"
#include "GeneticIndividual.hpp"

/**
 * @brief Evolves fixed-size generations through an ask--tell protocol.
 * @tparam Genome Copyable candidate representation.
 * @tparam CrossoverOperator Callable producing a `Genome` from two parents and `std::mt19937_64&`.
 * @tparam MutationOperator Callable mutating a `Genome&` with `std::mt19937_64&`.
 *
 * Successful individuals are eligible for elitism and parent selection.
 * Tournament participants are sampled with replacement from that set. A
 * participant is then selected proportionally to its non-negative fitness, or
 * uniformly when every tournament weight is zero.
 *
 * @see genetic_search_chapter
 */
template< typename Genome, typename CrossoverOperator, typename MutationOperator >
class GeneticAlgorithmEngine
{
  public:
  /**
   * @brief Creates generation zero from an explicit population.
   * @param[in] configuration Population, elitism, tournament, and seed settings.
   * @param[in] initial_population Genomes moved into generation zero in supplied order.
   * @param[in] crossover Crossover operator owned by the engine.
   * @param[in] mutation Mutation operator owned by the engine.
   * @throws std::invalid_argument If population size is below two, the initial
   *         cardinality differs from it, elite count is not smaller than it, or
   *         tournament size lies outside `[1, population_size]`.
   */
  GeneticAlgorithmEngine( GeneticAlgorithmConfiguration configuration, std::vector< Genome > initial_population,
                          CrossoverOperator crossover, MutationOperator mutation ) :
      configuration_( std::move( configuration ) ), crossover_( std::move( crossover ) ),
      mutation_( std::move( mutation ) ), generator_( configuration_.random_seed )
  {
    validateConfiguration( initial_population.size() );

    population_.reserve( initial_population.size() );
    for ( Genome &genome : initial_population )
    {
      population_.push_back( GeneticIndividual< Genome >{ std::move( genome ), std::nullopt } );
    }
  }

  /**
   * @brief Returns the complete current generation and opens a fitness transaction.
   * @return Genome copies in population order.
   * @throws std::logic_error If feedback for a preceding generation is pending.
   * @post A matching `tellGeneration()` is required before the next call.
   */
  std::vector< Genome > askGeneration()
  {
    if ( waiting_for_fitness_ )
    {
      throw std::logic_error( "GeneticAlgorithmEngine: ask called while waiting for fitness" );
    }

    std::vector< Genome > generation;
    generation.reserve( population_.size() );
    for ( const GeneticIndividual< Genome > &individual : population_ )
    {
      generation.push_back( individual.genome );
    }

    waiting_for_fitness_ = true;
    return generation;
  }

  /**
   * @brief Applies ordered fitness feedback and creates the next generation.
   *
   * Successful negative fitness values are clamped to zero. Failed values are
   * set to zero and excluded from reproduction. Up to `elite_count` successful
   * individuals are copied by descending fitness; crossover and mutation fill
   * the remaining population.
   *
   * @param[in] fitness One entry per pending genome, in population order.
   * @throws std::logic_error If no generation is pending.
   * @throws std::invalid_argument If cardinality differs from the population or
   *         a successful fitness is non-finite.
   * @throws std::runtime_error If the generation has no successful evaluation.
   * @post On success, `generation()` is incremented and a new generation may be requested.
   */
  void tellGeneration( const std::vector< GeneticFitness > &fitness )
  {
    if ( !waiting_for_fitness_ )
    {
      throw std::logic_error( "GeneticAlgorithmEngine: tell called without a pending generation" );
    }
    if ( fitness.size() != population_.size() )
    {
      throw std::invalid_argument( "GeneticAlgorithmEngine: fitness count does not match population size" );
    }

    std::size_t successful_count = 0;
    for ( std::size_t index = 0; index < fitness.size(); ++index )
    {
      GeneticFitness normalized_fitness = fitness.at( index );
      if ( normalized_fitness.status == GeneticEvaluationStatus::Succeeded )
      {
        if ( !std::isfinite( normalized_fitness.value ) )
        {
          throw std::invalid_argument( "GeneticAlgorithmEngine: successful fitness must be finite" );
        }

        normalized_fitness.value = std::max( 0.0, normalized_fitness.value );
        ++successful_count;
      }
      else
      {
        normalized_fitness.value = 0.0;
      }

      population_.at( index ).fitness = normalized_fitness;
    }

    if ( successful_count == 0 )
    {
      throw std::runtime_error( "GeneticAlgorithmEngine: generation contains no successful evaluations" );
    }

    createNextGeneration();
    ++generation_;
    waiting_for_fitness_ = false;
  }

  /** @brief Returns the number of completed evolution transitions. @return Zero before the first successful `tellGeneration()`. */
  std::size_t generation() const { return generation_; }

  private:
  /**
   * @brief Checks population-wide configuration invariants.
   * @param[in] initial_population_size Cardinality supplied to the constructor.
   * @throws std::invalid_argument If a population, elitism, or tournament invariant fails.
   */
  void validateConfiguration( std::size_t initial_population_size ) const
  {
    if ( configuration_.population_size < 2 )
    {
      throw std::invalid_argument( "GeneticAlgorithmEngine: population size must be at least two" );
    }
    if ( initial_population_size != configuration_.population_size )
    {
      throw std::invalid_argument( "GeneticAlgorithmEngine: initial population size does not match configuration" );
    }
    if ( configuration_.elite_count >= configuration_.population_size )
    {
      throw std::invalid_argument( "GeneticAlgorithmEngine: elite count must be smaller than population size" );
    }
    if ( configuration_.tournament_size == 0 ||
         configuration_.tournament_size > configuration_.population_size )
    {
      throw std::invalid_argument(
        "GeneticAlgorithmEngine: tournament size must be between one and population size" );
    }
  }

  /**
   * @brief Collects reproduction-eligible population positions.
   * @return Indices whose optional fitness is present and marked successful, in population order.
   */
  std::vector< std::size_t > successfulIndices() const
  {
    std::vector< std::size_t > indices;
    indices.reserve( population_.size() );

    for ( std::size_t index = 0; index < population_.size(); ++index )
    {
      const std::optional< GeneticFitness > &fitness = population_.at( index ).fitness;
      if ( fitness.has_value() && fitness->status == GeneticEvaluationStatus::Succeeded )
      {
        indices.push_back( index );
      }
    }

    return indices;
  }

  /**
   * @brief Selects one parent through a sampled tournament.
   * @return Reference into `population_`, valid until the population is replaced.
   * @throws std::logic_error If no successful individual exists.
   */
  const GeneticIndividual< Genome > &selectParent()
  {
    const std::vector< std::size_t > successful_indices = successfulIndices();
    if ( successful_indices.empty() )
    {
      throw std::logic_error( "GeneticAlgorithmEngine: parent selection requires a successful individual" );
    }

    std::uniform_int_distribution< std::size_t > individual_distribution( 0, successful_indices.size() - 1 );
    std::vector< std::size_t > tournament;
    std::vector< double > weights;
    tournament.reserve( configuration_.tournament_size );
    weights.reserve( configuration_.tournament_size );

    for ( std::size_t participant = 0; participant < configuration_.tournament_size; ++participant )
    {
      const std::size_t population_index = successful_indices.at( individual_distribution( generator_ ) );
      tournament.push_back( population_index );
      weights.push_back( population_.at( population_index ).fitness->value );
    }

    const double total_weight = std::accumulate( weights.begin(), weights.end(), 0.0 );
    std::size_t selected_participant = 0;
    if ( total_weight == 0.0 )
    {
      std::uniform_int_distribution< std::size_t > participant_distribution( 0, tournament.size() - 1 );
      selected_participant = participant_distribution( generator_ );
    }
    else
    {
      std::discrete_distribution< std::size_t > participant_distribution( weights.begin(), weights.end() );
      selected_participant = participant_distribution( generator_ );
    }

    return population_.at( tournament.at( selected_participant ) );
  }

  /**
   * @brief Builds the next population from elites and generated children.
   *
   * Successful genomes are sorted by descending fitness. The configured elite
   * prefix, limited by successful count, is copied. Each remaining child is
   * produced from two independently selected parents and then mutated.
   */
  void createNextGeneration()
  {
    std::vector< std::size_t > successful_indices = successfulIndices();
    std::sort( successful_indices.begin(), successful_indices.end(), [this]( std::size_t left, std::size_t right )
               { return population_.at( left ).fitness->value > population_.at( right ).fitness->value; } );

    std::vector< GeneticIndividual< Genome > > next_population;
    next_population.reserve( configuration_.population_size );

    const std::size_t copied_elite_count = std::min( configuration_.elite_count, successful_indices.size() );
    for ( std::size_t elite = 0; elite < copied_elite_count; ++elite )
    {
      next_population.push_back(
        GeneticIndividual< Genome >{ population_.at( successful_indices.at( elite ) ).genome, std::nullopt } );
    }

    while ( next_population.size() < configuration_.population_size )
    {
      const GeneticIndividual< Genome > &first_parent = selectParent();
      const GeneticIndividual< Genome > &second_parent = selectParent();
      Genome child = crossover_( first_parent.genome, second_parent.genome, generator_ );
      mutation_( child, generator_ );
      next_population.push_back( GeneticIndividual< Genome >{ std::move( child ), std::nullopt } );
    }

    population_ = std::move( next_population );
  }

  GeneticAlgorithmConfiguration configuration_; ///< Fixed population and selection settings.
  std::vector< GeneticIndividual< Genome > > population_; ///< Current ordered generation and its optional feedback.
  CrossoverOperator crossover_; ///< Owned child-construction operator.
  MutationOperator mutation_; ///< Owned post-crossover mutation operator.
  std::mt19937_64 generator_; ///< Persistent random stream seeded from the configuration.
  std::size_t generation_ = 0; ///< Number of successfully completed generations.
  bool waiting_for_fitness_ = false; ///< Whether `askGeneration()` has opened a pending transaction.
};

/** @} */

#endif // !GENETIC_ALGORITHM_ENGINE_H
