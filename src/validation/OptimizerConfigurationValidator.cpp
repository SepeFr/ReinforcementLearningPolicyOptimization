#include "validation/OptimizerConfigurationValidator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include "FeedForwardNetworkConfiguration.hpp"
#include "InitializationConfiguration.hpp"
#include "MethodHyperparameters.hpp"
#include "NeighborhoodConfiguration.hpp"
#include "OptimizerConfiguration.hpp"
#include "StoppingConfiguration.hpp"
#include "validation/InvalidConfigurationError.hpp"
#include "validation/PolicyConfigurationValidator.hpp"

namespace
{
  bool isFinitePositive( double value ) { return std::isfinite( value ) && value > 0.0; }

  bool isFiniteNonNegative( double value ) { return std::isfinite( value ) && value >= 0.0; }

  bool isInOpenUnitInterval( double value )
  {
    return std::isfinite( value ) && value > 0.0 && value < 1.0;
  }

  bool isKnownInitializationType( FeedForwardInitializationType type )
  {
    switch ( type )
    {
      case FeedForwardInitializationType::FanInUniform:
      case FeedForwardInitializationType::XavierUniform:
        return true;
    }

    return false;
  }

  bool hasValidParameterVector( const Eigen::VectorXd &parameters, std::size_t parameter_count )
  {
    return parameters.size() == static_cast< Eigen::Index >( parameter_count ) && parameters.allFinite();
  }

  bool hasValidInitialization( const InitializationConfiguration &configuration, std::size_t parameter_count )
  {
    if ( configuration.valueless_by_exception() )
    {
      return false;
    }

    return std::visit(
      [parameter_count]( const auto &selected_configuration )
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, RandomInitializationConfiguration > )
        {
          const auto has_valid_bound = [parameter_count]( const std::optional< Eigen::VectorXd > &bound )
          {
            return !bound.has_value() || hasValidParameterVector( *bound, parameter_count );
          };

          if ( !has_valid_bound( selected_configuration.random_lower_bound ) ||
               !has_valid_bound( selected_configuration.random_upper_bound ) )
          {
            return false;
          }

          return !selected_configuration.random_lower_bound.has_value() ||
            !selected_configuration.random_upper_bound.has_value() ||
            ( selected_configuration.random_lower_bound->array() <=
              selected_configuration.random_upper_bound->array() )
              .all();
        }
        else if constexpr ( std::is_same_v< ConfigurationType, FeedForwardInitializationConfiguration > )
        {
          PolicyConfigurationValidator::validate( selected_configuration.network );
          return isKnownInitializationType( selected_configuration.type ) &&
            FeedForwardNetworkConfiguration::parameterCountFromConfiguration( selected_configuration.network ) ==
            parameter_count;
        }
        else if constexpr ( std::is_same_v< ConfigurationType, ProvidedInitializationConfiguration > )
        {
          return hasValidParameterVector( selected_configuration.provided_parameters, parameter_count );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, ProvidedInitializationVectorConfiguration > )
        {
          return !selected_configuration.provided_parameters.empty() &&
            std::all_of( selected_configuration.provided_parameters.begin(),
                         selected_configuration.provided_parameters.end(),
                         [parameter_count]( const Eigen::VectorXd &parameters )
                         { return hasValidParameterVector( parameters, parameter_count ); } );
        }
        else
        {
          return false;
        }
      },
      configuration );
  }

  bool hasFullAffineRank( const std::vector< Eigen::VectorXd > &vertices, std::size_t parameter_count )
  {
    if ( vertices.size() != parameter_count + 1 )
    {
      return false;
    }

    const Eigen::Index dimension = static_cast< Eigen::Index >( parameter_count );
    Eigen::MatrixXd difference_matrix( dimension, dimension );

    for ( Eigen::Index column = 0; column < dimension; ++column )
    {
      difference_matrix.col( column ) =
        vertices.at( static_cast< std::size_t >( column + 1 ) ) - vertices.front();
    }

    Eigen::FullPivLU< Eigen::MatrixXd > decomposition( difference_matrix );
    decomposition.setThreshold( std::sqrt( std::numeric_limits< double >::epsilon() ) );
    return decomposition.rank() == dimension;
  }

  bool hasValidNeighborhood( const NeighborhoodConfiguration &configuration )
  {
    if ( configuration.valueless_by_exception() )
    {
      return false;
    }

    return std::visit(
      []( const auto &selected_configuration )
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, GaussianNeighborhoodConfiguration > ||
                       std::is_same_v< ConfigurationType, LatinHypercubeGaussianNeighborhoodConfiguration > )
        {
          return std::isfinite( selected_configuration.perturbation_mean ) &&
            isFinitePositive( selected_configuration.standard_deviation );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, UniformNeighborhoodConfiguration > ||
                            std::is_same_v< ConfigurationType, CoordinateNeighborhoodConfiguration > ||
                            std::is_same_v< ConfigurationType, LatinHypercubeUniformNeighborhoodConfiguration > )
        {
          return std::isfinite( selected_configuration.lower_bound ) &&
            std::isfinite( selected_configuration.upper_bound ) &&
            selected_configuration.lower_bound <= selected_configuration.upper_bound;
        }
        else
        {
          return false;
        }
      },
      configuration );
  }

  bool hasValidCoolingFunction( const CoolingFunctionConfiguration &configuration, double initial_temperature )
  {
    if ( configuration.valueless_by_exception() )
    {
      return false;
    }

    return std::visit(
      [initial_temperature]( const auto &selected_configuration )
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, GeometricCoolingConfiguration > )
        {
          return isInOpenUnitInterval( selected_configuration.alpha );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, LinearCoolingConfiguration > )
        {
          return isFinitePositive( selected_configuration.beta ) &&
            isFiniteNonNegative( selected_configuration.minimum_temperature ) &&
            selected_configuration.minimum_temperature < initial_temperature;
        }
        else if constexpr ( std::is_same_v< ConfigurationType, LogarithmicCoolingConfiguration > )
        {
          return isFinitePositive( selected_configuration.c ) && isFinitePositive( selected_configuration.d );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, ReciprocalCoolingConfiguration > )
        {
          return isFinitePositive( selected_configuration.b );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, ExponentialCoolingConfiguration > )
        {
          return isFinitePositive( selected_configuration.beta );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, InverseLogarithmicCoolingConfiguration > )
        {
          return std::isfinite( selected_configuration.dimension ) && selected_configuration.dimension > 1.0;
        }
        else
        {
          return false;
        }
      },
      configuration );
  }

  bool coolingCanReachTemperature( const SimulatedAnnealingHyperparameters &hyperparameters )
  {
    if ( hyperparameters.minimal_temperature <= 0.0 )
    {
      return false;
    }

    if ( const auto *linear = std::get_if< LinearCoolingConfiguration >( &hyperparameters.cooling_function ) )
    {
      return linear->minimum_temperature < hyperparameters.minimal_temperature;
    }

    return true;
  }

  bool hasValidNelderMeadStopping( const NelderMeadStoppingConfiguration &configuration )
  {
    if ( configuration.valueless_by_exception() )
    {
      return false;
    }

    return std::visit(
      []( const auto &selected_configuration )
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, DispersionStoppingConfiguration > )
        {
          return isFiniteNonNegative( selected_configuration.x_absolute_tolerance ) &&
            isFiniteNonNegative( selected_configuration.f_absolute_tolerance );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, RelativeVariationStoppingConfiguration > ||
                            std::is_same_v< ConfigurationType, StandardDeviationStoppingConfiguration > )
        {
          return isFiniteNonNegative( selected_configuration.tolerance );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, HeightDifferenceStoppingConfiguration > )
        {
          const double absolute_tolerance = selected_configuration.absolute_tolerance;
          const bool valid_absolute_tolerance = std::isfinite( absolute_tolerance ) ||
            ( std::isinf( absolute_tolerance ) && absolute_tolerance < 0.0 );

          return isFiniteNonNegative( selected_configuration.convergence_tolerance ) &&
            valid_absolute_tolerance;
        }
        else if constexpr ( std::is_same_v< ConfigurationType, RelativeFunctionToleranceStoppingConfiguration > )
        {
          return isFiniteNonNegative( selected_configuration.relative_tolerance );
        }
        else
        {
          return false;
        }
      },
      configuration );
  }

  bool hasValidNelderMeadInitialization( const NelderMeadHyperparameters &hyperparameters,
                                         const InitializationConfiguration &initialization,
                                         std::size_t parameter_count )
  {
    const auto *provided_initialization =
      std::get_if< ProvidedInitializationVectorConfiguration >( &initialization );
    const bool uses_provided_simplex =
      std::holds_alternative< ProvidedSimplexConfiguration >( hyperparameters.initialization );

    if ( uses_provided_simplex != ( provided_initialization != nullptr ) ||
         hyperparameters.initialization.valueless_by_exception() )
    {
      return false;
    }

    return std::visit(
      [&]( const auto &selected_configuration )
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, ClassicalLocalSimplexConfiguration > )
        {
          return true;
        }
        else if constexpr ( std::is_same_v< ConfigurationType, LatinHypercubeLocalSimplexConfiguration > )
        {
          return selected_configuration.maximum_attempts > 0;
        }
        else if constexpr ( std::is_same_v< ConfigurationType, LatinHypercubeGlobalSimplexConfiguration > )
        {
          const Eigen::Index dimension = static_cast< Eigen::Index >( parameter_count );
          return selected_configuration.maximum_attempts > 0 &&
            selected_configuration.lower_bound.size() == dimension &&
            selected_configuration.upper_bound.size() == dimension &&
            selected_configuration.lower_bound.allFinite() &&
            selected_configuration.upper_bound.allFinite() &&
            ( selected_configuration.lower_bound.array() < selected_configuration.upper_bound.array() ).all();
        }
        else if constexpr ( std::is_same_v< ConfigurationType, ProvidedSimplexConfiguration > )
        {
          return provided_initialization != nullptr &&
            hasFullAffineRank( provided_initialization->provided_parameters, parameter_count );
        }
        else
        {
          return false;
        }
      },
      hyperparameters.initialization );
  }

  bool hasValidNelderMeadHyperparameters( const NelderMeadHyperparameters &hyperparameters,
                                          const InitializationConfiguration &initialization,
                                          std::size_t parameter_count )
  {
    return isFinitePositive( hyperparameters.initial_simplex_scale ) &&
      std::isfinite( hyperparameters.reflection_coefficient ) &&
      std::isfinite( hyperparameters.expansion_coefficient ) &&
      std::isfinite( hyperparameters.inside_contraction_coefficient ) &&
      std::isfinite( hyperparameters.outside_contraction_coefficient ) &&
      std::isfinite( hyperparameters.shrink_coefficient ) &&
      hasValidNelderMeadStopping( hyperparameters.stopping_criterion ) &&
      hasValidNelderMeadInitialization( hyperparameters, initialization, parameter_count );
  }

  bool hasValidRandomMeshSearch( const RandomMeshGPSSearchConfiguration &configuration )
  {
    return configuration.number_of_points == 0 || configuration.l1_radius > 0;
  }

  bool hasValidSurrogateModel( const SurrogateModelConfiguration &configuration )
  {
    if ( configuration.valueless_by_exception() )
    {
      return false;
    }

    return std::visit(
      []( const auto &selected_configuration )
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, LocalQuadraticSurrogateConfiguration > )
        {
          return isFiniteNonNegative( selected_configuration.curvature_regularization ) &&
            isFinitePositive( selected_configuration.local_radius_multiplier );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, RBFSurrogateConfiguration > )
        {
          const bool known_function = selected_configuration.function == RBFFunction::Cubic ||
            selected_configuration.function == RBFFunction::Linear;

          return known_function && isFiniteNonNegative( selected_configuration.regularization ) &&
            isFinitePositive( selected_configuration.local_radius_multiplier ) &&
            selected_configuration.maximum_coreset_size > 0;
        }
        else
        {
          return false;
        }
      },
      configuration );
  }

  bool hasValidGPSSearch( const GPSSearchConfiguration &configuration )
  {
    if ( configuration.valueless_by_exception() )
    {
      return false;
    }

    return std::visit(
      []( const auto &selected_configuration )
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, EmptyGPSSearchConfiguration > )
        {
          return true;
        }
        else if constexpr ( std::is_same_v< ConfigurationType, RandomMeshGPSSearchConfiguration > )
        {
          return hasValidRandomMeshSearch( selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, LatinHypercubeMeshGPSSearchConfiguration > )
        {
          return ( selected_configuration.number_of_points == 0 ||
                   ( selected_configuration.l1_radius > 0 && selected_configuration.maximum_batches > 0 ) );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, SuccessfulDirectionGPSSearchConfiguration > )
        {
          return hasValidRandomMeshSearch( selected_configuration.initial_search );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, SurrogateGPSSearchConfiguration > )
        {
          return hasValidRandomMeshSearch( selected_configuration.random_points_strategy ) &&
            std::isfinite( selected_configuration.selected_fraction ) &&
            selected_configuration.selected_fraction > 0.0 && selected_configuration.selected_fraction <= 1.0 &&
            hasValidSurrogateModel( selected_configuration.model );
        }
        else
        {
          return false;
        }
      },
      configuration );
  }

  bool hasValidGPSHyperparameters( const GeneralizedPatternSearchHyperparameters &hyperparameters,
                                   std::size_t parameter_count )
  {
    if ( !isFinitePositive( hyperparameters.initial_mesh_size ) ||
         !isInOpenUnitInterval( hyperparameters.mesh_size_adjustment ) ||
         !isFiniteNonNegative( hyperparameters.stopping_mesh_size ) ||
         hyperparameters.positive_basis.valueless_by_exception() || hyperparameters.poll.valueless_by_exception() ||
         !hasValidGPSSearch( hyperparameters.search ) )
    {
      return false;
    }

    const bool known_positive_basis =
      std::holds_alternative< MinimalPositiveBasisConfiguration >( hyperparameters.positive_basis ) ||
      std::holds_alternative< SymmetricPositiveBasisConfiguration >( hyperparameters.positive_basis );
    const bool known_poll = std::holds_alternative< CompleteGPSPollConfiguration >( hyperparameters.poll );

    if ( !known_positive_basis || !known_poll )
    {
      return false;
    }
    if ( !hyperparameters.generating_matrix.has_value() )
    {
      return true;
    }

    const Eigen::MatrixXd &generating_matrix = *hyperparameters.generating_matrix;
    const Eigen::Index dimension = static_cast< Eigen::Index >( parameter_count );
    return generating_matrix.rows() == dimension && generating_matrix.cols() == dimension &&
      generating_matrix.allFinite() && Eigen::FullPivLU< Eigen::MatrixXd >( generating_matrix ).isInvertible();
  }

  bool hasValidStochasticGPSStopping( const RinottRankingSelectionGPSHyperparameters &hyperparameters )
  {
    if ( hyperparameters.stopping_criterion.valueless_by_exception() )
    {
      return false;
    }

    return std::visit(
      [&]( const auto &selected_configuration )
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, NoStochasticGPSStoppingConfiguration > )
        {
          return true;
        }
        else if constexpr ( std::is_same_v< ConfigurationType, StochasticGPSMeshSizeStoppingConfiguration > )
        {
          return hyperparameters.stopping_mesh_size > 0.0;
        }
        else if constexpr ( std::is_same_v< ConfigurationType, StochasticGPSSignificanceStoppingConfiguration > )
        {
          return isInOpenUnitInterval( selected_configuration.tolerance );
        }
        else if constexpr (
          std::is_same_v< ConfigurationType, StochasticGPSNoiseToIndifferenceStoppingConfiguration > )
        {
          return isFinitePositive( selected_configuration.threshold );
        }
        else
        {
          return false;
        }
      },
      hyperparameters.stopping_criterion );
  }

  bool supportsInitialization( const MethodHyperparameters &hyperparameters,
                               const InitializationConfiguration &initialization )
  {
    const bool has_initialization_vector =
      std::holds_alternative< ProvidedInitializationVectorConfiguration >( initialization );

    return std::visit(
      [&]( const auto &selected_hyperparameters )
      {
        using Hyperparameters = std::decay_t< decltype( selected_hyperparameters ) >;

        if constexpr ( std::is_same_v< Hyperparameters, HillClimbingHyperparameters > ||
                       std::is_same_v< Hyperparameters, SimulatedAnnealingHyperparameters > ||
                       std::is_same_v< Hyperparameters, GeneralizedPatternSearchHyperparameters > ||
                       std::is_same_v< Hyperparameters, RinottRankingSelectionGPSHyperparameters > ||
                       std::is_same_v< Hyperparameters, CMAESHyperparameters > ||
                       std::is_same_v< Hyperparameters, PSOHyperparameters > )
        {
          return !has_initialization_vector;
        }
        else if constexpr ( std::is_same_v< Hyperparameters, SPSAHyperparameters > )
        {
          return std::holds_alternative< RandomInitializationConfiguration >( initialization ) ||
            std::holds_alternative< ProvidedInitializationConfiguration >( initialization );
        }
        else
        {
          return true;
        }
      },
      hyperparameters );
  }

  bool hasValidMethodConfiguration( const OptimizerConfiguration &configuration, std::size_t parameter_count )
  {
    if ( configuration.hyperparameters.valueless_by_exception() ||
         !supportsInitialization( configuration.hyperparameters, configuration.initialization ) )
    {
      return false;
    }

    return std::visit(
      [&]( const auto &hyperparameters )
      {
        using Hyperparameters = std::decay_t< decltype( hyperparameters ) >;

        if constexpr ( std::is_same_v< Hyperparameters, HillClimbingHyperparameters > )
        {
          return hyperparameters.number_of_neighbors > 0 &&
            isFinitePositive( hyperparameters.neighborhood_scale ) &&
            hasValidNeighborhood( hyperparameters.neighborhood );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, SimulatedAnnealingHyperparameters > )
        {
          return isFinitePositive( hyperparameters.initial_temperature ) &&
            isFiniteNonNegative( hyperparameters.minimal_temperature ) &&
            isFinitePositive( hyperparameters.neighborhood_scale ) &&
            hasValidNeighborhood( hyperparameters.neighborhood ) &&
            hasValidCoolingFunction( hyperparameters.cooling_function, hyperparameters.initial_temperature );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, NelderMeadHyperparameters > ||
                            std::is_same_v< Hyperparameters, StochasticHeuristicNelderMeadHyperparameters > )
        {
          return hasValidNelderMeadHyperparameters( hyperparameters, configuration.initialization,
                                                    parameter_count );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, ChangStochasticNelderMeadHyperparameters > )
        {
          const Eigen::Index dimension = static_cast< Eigen::Index >( parameter_count );
          const bool valid_bounds = hyperparameters.lower_bound.size() == dimension &&
            hyperparameters.upper_bound.size() == dimension &&
            hyperparameters.lower_bound.allFinite() && hyperparameters.upper_bound.allFinite() &&
            ( hyperparameters.lower_bound.array() < hyperparameters.upper_bound.array() ).all();

          return hasValidNelderMeadHyperparameters( hyperparameters, configuration.initialization,
                                                    parameter_count ) &&
            hyperparameters.minimum_sample_size > 0 &&
            isInOpenUnitInterval( hyperparameters.global_search_probability ) && valid_bounds;
        }
        else if constexpr ( std::is_same_v< Hyperparameters, GeneralizedPatternSearchHyperparameters > )
        {
          return hasValidGPSHyperparameters( hyperparameters, parameter_count );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, RinottRankingSelectionGPSHyperparameters > )
        {
          return hasValidGPSHyperparameters( hyperparameters, parameter_count ) &&
            hyperparameters.initial_sample_size >= 2 &&
            isInOpenUnitInterval( hyperparameters.initial_significance ) &&
            isFinitePositive( hyperparameters.initial_indifference ) &&
            isInOpenUnitInterval( hyperparameters.significance_decay ) &&
            isInOpenUnitInterval( hyperparameters.indifference_decay ) &&
            hasValidStochasticGPSStopping( hyperparameters );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, OpenAIESHyperparameters > )
        {
          return hyperparameters.population_size >= 2 && hyperparameters.population_size % 2 == 0 &&
            isFinitePositive( hyperparameters.learning_rate ) && isFinitePositive( hyperparameters.noise_scale ) &&
            std::isfinite( hyperparameters.beta_1 ) && hyperparameters.beta_1 >= 0.0 && hyperparameters.beta_1 < 1.0 &&
            std::isfinite( hyperparameters.beta_2 ) && hyperparameters.beta_2 >= 0.0 && hyperparameters.beta_2 < 1.0 &&
            isFinitePositive( hyperparameters.epsilon ) && std::isfinite( hyperparameters.weight_decay ) &&
            hyperparameters.weight_decay >= 0.0 &&
            ( !hyperparameters.adapt_noise_scale ||
              ( std::isfinite( hyperparameters.target_success_rate ) &&
                hyperparameters.target_success_rate > 0.0 && hyperparameters.target_success_rate < 1.0 &&
                std::isfinite( hyperparameters.noise_scale_increase_factor ) &&
                hyperparameters.noise_scale_increase_factor > 1.0 &&
                std::isfinite( hyperparameters.noise_scale_decrease_factor ) &&
                hyperparameters.noise_scale_decrease_factor > 0.0 &&
                hyperparameters.noise_scale_decrease_factor < 1.0 &&
                isFinitePositive( hyperparameters.minimum_noise_scale ) &&
                std::isfinite( hyperparameters.maximum_noise_scale ) &&
                hyperparameters.maximum_noise_scale >= hyperparameters.minimum_noise_scale &&
                hyperparameters.noise_scale >= hyperparameters.minimum_noise_scale &&
                hyperparameters.noise_scale <= hyperparameters.maximum_noise_scale ) );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, CMAESHyperparameters > )
        {
          return hyperparameters.population_size >= 2 && isFinitePositive( hyperparameters.initial_sigma );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, PSOHyperparameters > )
        {
          return hyperparameters.population_size > 0 &&
            isFiniteNonNegative( hyperparameters.initial_inertia_weight ) &&
            isFiniteNonNegative( hyperparameters.cognitive_coefficient ) &&
            isFiniteNonNegative( hyperparameters.social_coefficient ) &&
            isFinitePositive( hyperparameters.initial_velocity_scale );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, SPSAHyperparameters > )
        {
          return isFinitePositive( hyperparameters.perturbation_magnitude ) &&
            isFinitePositive( hyperparameters.step_size );
        }
        else
        {
          return false;
        }
      },
      configuration.hyperparameters );
  }

  bool hasValidStoppingConfiguration( const std::vector< StoppingConfiguration > &configurations )
  {
    std::array< bool, std::variant_size_v< StoppingConfiguration > > configured_types{};

    for ( const StoppingConfiguration &configuration : configurations )
    {
      if ( configuration.valueless_by_exception() || configured_types.at( configuration.index() ) )
      {
        return false;
      }
      configured_types.at( configuration.index() ) = true;

      const bool valid = std::visit(
        []( const auto &selected_configuration )
        {
          using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

          if constexpr ( std::is_same_v< ConfigurationType, MaximumIterationsStoppingConfiguration > )
          {
            return selected_configuration.maximum_iterations > 0;
          }
          else if constexpr ( std::is_same_v< ConfigurationType, MaximumEvaluationsStoppingConfiguration > )
          {
            return selected_configuration.maximum_evaluations > 0;
          }
          else if constexpr ( std::is_same_v< ConfigurationType, NoImprovementStoppingConfiguration > )
          {
            return selected_configuration.maximum_iterations_without_improvement > 0 &&
              isFiniteNonNegative( selected_configuration.threshold );
          }
          else if constexpr ( std::is_same_v< ConfigurationType, TargetValueStoppingConfiguration > )
          {
            return std::isfinite( selected_configuration.target_value );
          }
          else
          {
            return false;
          }
        },
        configuration );

      if ( !valid )
      {
        return false;
      }
    }

    return true;
  }

  bool hasInternalStoppingCriterion( const MethodHyperparameters &hyperparameters )
  {
    return std::visit(
      []( const auto &selected_hyperparameters )
      {
        using Hyperparameters = std::decay_t< decltype( selected_hyperparameters ) >;

        if constexpr ( std::is_same_v< Hyperparameters, SimulatedAnnealingHyperparameters > )
        {
          return coolingCanReachTemperature( selected_hyperparameters );
        }
        else if constexpr ( std::is_same_v< Hyperparameters, NelderMeadHyperparameters > ||
                            std::is_same_v< Hyperparameters, StochasticHeuristicNelderMeadHyperparameters > ||
                            std::is_same_v< Hyperparameters, ChangStochasticNelderMeadHyperparameters > )
        {
          return true;
        }
        else if constexpr ( std::is_same_v< Hyperparameters, GeneralizedPatternSearchHyperparameters > )
        {
          return selected_hyperparameters.stopping_mesh_size > 0.0;
        }
        else if constexpr ( std::is_same_v< Hyperparameters, RinottRankingSelectionGPSHyperparameters > )
        {
          return !std::holds_alternative< NoStochasticGPSStoppingConfiguration >(
            selected_hyperparameters.stopping_criterion );
        }
        else
        {
          return false;
        }
      },
      hyperparameters );
  }
} // namespace

void OptimizerConfigurationValidator::validate( const OptimizerConfiguration &configuration,
                                                 std::size_t parameter_count )
{
  if ( parameter_count == 0 )
  {
    throw InvalidConfigurationError( "OptimizerConfigurationValidator: parameter count must be greater than zero" );
  }
  if ( !std::in_range< Eigen::Index >( parameter_count ) )
  {
    throw InvalidConfigurationError(
      "OptimizerConfigurationValidator: parameter count is not representable as Eigen::Index" );
  }

  bool valid_initialization = false;
  try
  {
    valid_initialization = hasValidInitialization( configuration.initialization, parameter_count );
  }
  catch ( const std::overflow_error &error )
  {
    throw InvalidConfigurationError(
      std::string( "OptimizerConfigurationValidator: invalid initialization configuration: " ) + error.what() );
  }

  if ( !valid_initialization )
  {
    throw InvalidConfigurationError( "OptimizerConfigurationValidator: invalid initialization configuration" );
  }
  if ( !hasValidStoppingConfiguration( configuration.stopping ) )
  {
    throw InvalidConfigurationError( "OptimizerConfigurationValidator: invalid stopping configuration" );
  }
  if ( !hasValidMethodConfiguration( configuration, parameter_count ) )
  {
    throw InvalidConfigurationError( "OptimizerConfigurationValidator: invalid method configuration" );
  }
  if ( configuration.stopping.empty() && !hasInternalStoppingCriterion( configuration.hyperparameters ) )
  {
    throw InvalidConfigurationError(
      "OptimizerConfigurationValidator: at least one stopping criterion is required" );
  }
}
