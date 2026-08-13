#ifndef INITIAL_PARAMETERS_STRATEGY_H
#define INITIAL_PARAMETERS_STRATEGY_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include "InitializationConfiguration.hpp"

/** @brief Interface that supplies candidates used to initialize an optimization method. */
class InitialParametersStrategy
{
  public:
  /** @brief Enables destruction through the strategy interface. */
  virtual ~InitialParametersStrategy() = default;

  /** @brief Produces the next initial candidate. @return Owning parameter vector. */
  virtual Eigen::VectorXd generateInitialParameters() = 0;
  /** @brief Restores deterministic state before a new optimization run. */
  virtual void reset() {}
};

/**
 * @brief Samples each initial coordinate independently from configured bounds.
 *
 * Two-sided coordinates use a uniform distribution. A one-sided coordinate is
 * its bound plus or minus the absolute value of a standard normal sample.
 * Unbounded coordinates use a uniform distribution on \f$[-1,1]\f$.
 */
class RandomInitialization : public InitialParametersStrategy
{
  public:
  /**
   * @brief Validates dimensions and initializes the random engine.
   * @param[in] parameter_count Number of coordinates generated per candidate.
   * @param[in] configuration Seed and optional bound vectors, copied into the strategy.
   * @throws std::invalid_argument If the count is zero or a bound length differs from it.
   * @throws std::overflow_error If the count cannot be represented as Eigen::Index.
   */
  RandomInitialization( std::size_t parameter_count,
                        RandomInitializationConfiguration configuration = RandomInitializationConfiguration{} ) :
      parameter_count_( parameter_count ), configuration_( std::move( configuration ) ),
      generator_( configuration_.random_seed )
  {
    if ( parameter_count_ == 0 )
    {
      throw std::invalid_argument( "RandomInitialization: parameter count must be greater than zero" );
    }
    if ( !std::in_range< Eigen::Index >( parameter_count_ ) )
    {
      throw std::overflow_error( "RandomInitialization: parameter count is not representable as Eigen::Index" );
    }
    if ( configuration_.random_lower_bound.has_value() &&
         configuration_.random_lower_bound->size() != static_cast< Eigen::Index >( parameter_count_ ) )
    {
      throw std::invalid_argument( "RandomInitialization: lower bound size does not match parameter count" );
    }
    if ( configuration_.random_upper_bound.has_value() &&
         configuration_.random_upper_bound->size() != static_cast< Eigen::Index >( parameter_count_ ) )
    {
      throw std::invalid_argument( "RandomInitialization: upper bound size does not match parameter count" );
    }
  }

  /**
   * @brief Samples one candidate according to each coordinate's bound state.
   * @return Vector containing parameter_count independently generated values.
   * @throws std::invalid_argument If a configured lower coordinate exceeds its upper coordinate.
   */
  Eigen::VectorXd generateInitialParameters() override
  {
    Eigen::VectorXd parameters( static_cast< Eigen::Index >( parameter_count_ ) );
    std::normal_distribution< double > one_sided_perturbation( 0.0, 1.0 );
    std::uniform_real_distribution< double > unbounded_distribution( -1.0, 1.0 );

    for ( Eigen::Index index = 0; index < parameters.size(); index++ )
    {
      const bool has_lower_bound = configuration_.random_lower_bound.has_value();
      const bool has_upper_bound = configuration_.random_upper_bound.has_value();

      if ( has_lower_bound && has_upper_bound )
      {
        const double lower_bound = configuration_.random_lower_bound->coeff( index );
        const double upper_bound = configuration_.random_upper_bound->coeff( index );
        if ( lower_bound > upper_bound )
        {
          throw std::invalid_argument( "RandomInitialization: lower bound cannot exceed upper bound" );
        }
        std::uniform_real_distribution< double > distribution( lower_bound, upper_bound );
        parameters[index] = distribution( generator_ );
      }
      else if ( has_lower_bound )
      {
        parameters[index] =
          configuration_.random_lower_bound->coeff( index ) + std::abs( one_sided_perturbation( generator_ ) );
      }
      else if ( has_upper_bound )
      {
        parameters[index] =
          configuration_.random_upper_bound->coeff( index ) - std::abs( one_sided_perturbation( generator_ ) );
      }
      else
      {
        parameters[index] = unbounded_distribution( generator_ );
      }
    }

    return parameters;
  }

  /** @brief Reseeds the engine with RandomInitializationConfiguration::random_seed. */
  void reset() override { generator_.seed( configuration_.random_seed ); }

  private:
  /** Candidate cardinality. */
  std::size_t parameter_count_;
  /** Copied seed and bound configuration. */
  RandomInitializationConfiguration configuration_;
  /** Deterministic 64-bit random engine. */
  std::mt19937_64 generator_;
};

/**
 * @brief Applies fan-in or Xavier uniform bounds to canonical network weights.
 *
 * Bias coordinates are fixed to zero. Layer slices follow FeedForwardNetwork's
 * serialization order.
 */
class FeedForwardInitialization final : public RandomInitialization
{
  public:
  /**
   * @brief Constructs layer-specific random bounds.
   * @param[in] parameter_count Expected canonical network parameter count.
   * @param[in] configuration Network layout, interval formula, and seed.
   * @throws std::invalid_argument If a layer width is zero or the expected
   * count disagrees with the derived bound-vector length.
   * @throws std::overflow_error If checked size arithmetic or Eigen conversion fails.
   */
  FeedForwardInitialization( std::size_t parameter_count,
                             const FeedForwardInitializationConfiguration &configuration ) :
      RandomInitialization( parameter_count, makeRandomConfiguration( configuration ) )
  {
  }

  private:
  /**
   * @brief Converts layer-aware settings to per-coordinate random bounds.
   * @param[in] configuration Network layout, weight formula, and seed.
   * @return Lower and upper vectors in canonical network parameter order.
   */
  static RandomInitializationConfiguration
  makeRandomConfiguration( const FeedForwardInitializationConfiguration &configuration )
  {
    const std::size_t parameter_count =
      FeedForwardNetworkConfiguration::parameterCountFromConfiguration( configuration.network );
    Eigen::VectorXd lower_bound( static_cast< Eigen::Index >( parameter_count ) );
    Eigen::VectorXd upper_bound( static_cast< Eigen::Index >( parameter_count ) );
    Eigen::Index offset = 0;

    const auto checked_add = []( std::size_t left, std::size_t right )
    {
      if ( right > std::numeric_limits< std::size_t >::max() - left )
      {
        throw std::overflow_error( "FeedForwardInitialization: layer size addition overflow" );
      }
      return left + right;
    };
    const auto checked_multiply = []( std::size_t left, std::size_t right )
    {
      if ( left != 0 && right > std::numeric_limits< std::size_t >::max() / left )
      {
        throw std::overflow_error( "FeedForwardInitialization: layer size multiplication overflow" );
      }
      return left * right;
    };

    const auto append_layer = [&]( std::size_t input_size, std::size_t output_size )
    {
      if ( input_size == 0 || output_size == 0 )
      {
        throw std::invalid_argument( "FeedForwardInitialization: layer sizes must be greater than zero" );
      }

      double bound = 0.0;
      switch ( configuration.type )
      {
        case FeedForwardInitializationType::FanInUniform:
          bound = 1.0 / std::sqrt( static_cast< double >( input_size ) );
          break;
        case FeedForwardInitializationType::XavierUniform:
          bound = std::sqrt( 6.0 / static_cast< double >( checked_add( input_size, output_size ) ) );
          break;
      }

      const std::size_t represented_weight_count = checked_multiply( input_size, output_size );
      if ( !std::in_range< Eigen::Index >( represented_weight_count ) )
      {
        throw std::overflow_error( "FeedForwardInitialization: weight count is not representable as Eigen::Index" );
      }
      const Eigen::Index weight_count = static_cast< Eigen::Index >( represented_weight_count );
      lower_bound.segment( offset, weight_count ).setConstant( -bound );
      upper_bound.segment( offset, weight_count ).setConstant( bound );
      offset += weight_count;

      if ( configuration.network.use_bias )
      {
        if ( !std::in_range< Eigen::Index >( output_size ) )
        {
          throw std::overflow_error( "FeedForwardInitialization: bias count is not representable as Eigen::Index" );
        }
        const Eigen::Index bias_count = static_cast< Eigen::Index >( output_size );
        lower_bound.segment( offset, bias_count ).setZero();
        upper_bound.segment( offset, bias_count ).setZero();
        offset += bias_count;
      }
    };

    std::size_t input_size = configuration.network.input_size;
    for ( const std::size_t output_size : configuration.network.hidden_layers )
    {
      append_layer( input_size, output_size );
      input_size = output_size;
    }
    append_layer( input_size, configuration.network.output_size );

    RandomInitializationConfiguration random_configuration;
    random_configuration.random_seed = configuration.random_seed;
    random_configuration.random_lower_bound = std::move( lower_bound );
    random_configuration.random_upper_bound = std::move( upper_bound );
    return random_configuration;
  }
};

/** @brief Returns the same caller-provided nonempty candidate on every request. */
class ProvidedInitialization : public InitialParametersStrategy
{
  public:
  /**
   * @brief Stores a fixed candidate.
   * @param[in] configuration Candidate copied into the strategy.
   * @throws std::invalid_argument If the candidate is empty.
   */
  explicit ProvidedInitialization( ProvidedInitializationConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
    if ( configuration_.provided_parameters.size() == 0 )
    {
      throw std::invalid_argument( "ProvidedInitialization: provided parameters cannot be empty" );
    }
  }

  /** @brief Returns the stored candidate. @return Owning copy of provided_parameters. */
  Eigen::VectorXd generateInitialParameters() override { return configuration_.provided_parameters; }

  private:
  /** Fixed candidate owned by the strategy. */
  ProvidedInitializationConfiguration configuration_;
};

/** @brief Cycles through a caller-provided sequence of initial candidates. */
class ProvidedInitializationVector : public InitialParametersStrategy
{
  public:
  /**
   * @brief Stores the candidate sequence.
   * @param[in] configuration Sequence copied into the strategy.
   * @throws std::invalid_argument If the sequence is empty.
   */
  explicit ProvidedInitializationVector( ProvidedInitializationVectorConfiguration configuration ) :
      configuration_( std::move( configuration ) )
  {
    if ( configuration_.provided_parameters.empty() )
    {
      throw std::invalid_argument( "ProvidedInitializationVector: provided parameters cannot be empty" );
    }
  }

  /**
   * @brief Returns the next candidate and advances the cyclic index.
   * @return Owning copy of the current sequence entry.
   */
  Eigen::VectorXd generateInitialParameters() override
  {
    const Eigen::VectorXd parameters = configuration_.provided_parameters.at( next_parameter_index_ );
    next_parameter_index_ = ( next_parameter_index_ + 1 ) % configuration_.provided_parameters.size();
    return parameters;
  }

  /** @brief Restores the next candidate to the first stored entry. */
  void reset() override { next_parameter_index_ = 0; }

  private:
  /** Candidate sequence owned by the strategy. */
  ProvidedInitializationVectorConfiguration configuration_;
  /** Index returned by the next generateInitialParameters() call. */
  std::size_t next_parameter_index_ = 0;
};

/** @} */

#endif // !INITIAL_PARAMETERS_STRATEGY_H
