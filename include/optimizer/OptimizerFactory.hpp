#ifndef OPTIMIZER_FACTORY_H
#define OPTIMIZER_FACTORY_H

/** @addtogroup optimizer_core_api
 * @{ */

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include "InitialParametersStrategy.hpp"
#include "InitializationConfiguration.hpp"
#include "MethodHyperparameters.hpp"
#include "NeighborhoodConfiguration.hpp"
#include "NeighborhoodStrategy.hpp"
#include "Optimizer.hpp"
#include "OptimizerConfiguration.hpp"
#include "OptimizerMethod.hpp"
#include "optimizations_methods/CMAESMethod.hpp"
#include "optimizations_methods/HillClimbingMethod.hpp"
#include "optimizations_methods/OpenAIESMethod.hpp"
#include "optimizations_methods/PSOMethod.hpp"
#include "optimizations_methods/SPSAMethod.hpp"
#include "optimizations_methods/gps/GPSMethod.hpp"
#include "optimizations_methods/gps/RinottRankingSelectionGPSMethod.hpp"
#include "optimizations_methods/nelder_mead/ChangStochasticNelderMeadMethod.hpp"
#include "optimizations_methods/nelder_mead/NelderMeadMethod.hpp"
#include "optimizations_methods/nelder_mead/StochasticHeuristicNelderMeadMethod.hpp"
#include "optimizations_methods/simulated_annealing/CoolingFunction.hpp"
#include "optimizations_methods/simulated_annealing/CoolingFunctionConfiguration.hpp"
#include "optimizations_methods/simulated_annealing/SimulatedAnnealingMethod.hpp"
#include "validation/InvalidConfigurationError.hpp"
#include "validation/OptimizerConfigurationValidator.hpp"

/**
 * @brief Validates an OptimizerConfiguration and assembles its owned runtime graph.
 *
 * Variant alternatives select the initialization, neighborhood, cooling, and
 * optimization classes. Created strategies are transferred to their owning
 * method or Optimizer.
 */
class OptimizerFactory
{
  private:
  /**
   * @brief Creates the selected initialization strategy.
   * @param[in] configuration Initialization variant.
   * @param[in] parameter_count Problem candidate cardinality.
   * @return Unique ownership of the selected strategy.
   */
  static std::unique_ptr< InitialParametersStrategy >
  createInitialParameters( const InitializationConfiguration &configuration, std::size_t parameter_count )
  {
    return std::visit(
      [&]( const auto &selected_configuration ) -> std::unique_ptr< InitialParametersStrategy >
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, RandomInitializationConfiguration > )
        {
          return std::make_unique< RandomInitialization >( parameter_count, selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, FeedForwardInitializationConfiguration > )
        {
          return std::make_unique< FeedForwardInitialization >( parameter_count, selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, ProvidedInitializationConfiguration > )
        {
          return std::make_unique< ProvidedInitialization >( selected_configuration );
        }
        else
        {
          return std::make_unique< ProvidedInitializationVector >( selected_configuration );
        }
      },
      configuration );
  }

  /**
   * @brief Creates the selected neighborhood strategy.
   * @param[in] configuration Neighborhood variant copied into the implementation.
   * @param[in] random_seed Seed shared by factory-created stochastic components.
   * @return Unique ownership of the selected strategy.
   */
  static std::unique_ptr< NeighborhoodStrategy > createNeighborhood( NeighborhoodConfiguration configuration,
                                                                     std::size_t random_seed )
  {
    std::unique_ptr< NeighborhoodStrategy > strategy = std::visit(
      [&]( const auto &configuration ) -> std::unique_ptr< NeighborhoodStrategy >
      {
        using T = std::decay_t< decltype( configuration ) >;

        if constexpr ( std::is_same_v< T, GaussianNeighborhoodConfiguration > )
        {
          return std::make_unique< GaussianNeighborhood >( configuration.perturbation_mean,
                                                           configuration.standard_deviation, random_seed );
        }
        else if constexpr ( std::is_same_v< T, UniformNeighborhoodConfiguration > )
        {
          return std::make_unique< UniformNeighborhood >( configuration.lower_bound, configuration.upper_bound,
                                                          random_seed );
        }
        else if constexpr ( std::is_same_v< T, CoordinateNeighborhoodConfiguration > )
        {
          return std::make_unique< CoordinateNeighborhood >( configuration.lower_bound, configuration.upper_bound,
                                                             random_seed );
        }
        else if constexpr ( std::is_same_v< T, LatinHypercubeUniformNeighborhoodConfiguration > )
        {
          return std::make_unique< LatinHypercubeUniformNeighborhood >( configuration, random_seed );
        }
        else
        {
          return std::make_unique< LatinHypercubeGaussianNeighborhood >( configuration, random_seed );
        }
      },
      configuration );
    return strategy;
  }

  /**
   * @brief Creates the selected simulated-annealing cooling function.
   * @param[in] configuration Cooling-function variant.
   * @param[in] initial_temperature Temperature at algorithmic time one.
   * @return Unique ownership of the selected cooling function.
   */
  static std::unique_ptr< CoolingFunction > createCoolingFunction( const CoolingFunctionConfiguration &configuration,
                                                                   double initial_temperature )
  {
    return std::visit(
      [&]( const auto &selected_configuration ) -> std::unique_ptr< CoolingFunction >
      {
        using ConfigurationType = std::decay_t< decltype( selected_configuration ) >;

        if constexpr ( std::is_same_v< ConfigurationType, GeometricCoolingConfiguration > )
        {
          return std::make_unique< GeometricCooling >( initial_temperature, selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, LinearCoolingConfiguration > )
        {
          return std::make_unique< LinearCooling >( initial_temperature, selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, LogarithmicCoolingConfiguration > )
        {
          return std::make_unique< LogarithmicCooling >( initial_temperature, selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, ReciprocalCoolingConfiguration > )
        {
          return std::make_unique< ReciprocalCooling >( initial_temperature, selected_configuration );
        }
        else if constexpr ( std::is_same_v< ConfigurationType, ExponentialCoolingConfiguration > )
        {
          return std::make_unique< ExponentialCooling >( initial_temperature, selected_configuration );
        }
        else
        {
          return std::make_unique< InverseLogarithmicCooling >( initial_temperature, selected_configuration );
        }
      },
      configuration );
  }

  public:
  /**
   * @brief Creates a validated custom optimizer using the standard runtime components.
   *
   * Extra arguments are forwarded after the four standard Optimizer constructor
   * arguments, allowing application-specific subclasses without coupling the
   * generic factory to them.
   */
  template< typename OptimizerType, typename... ExtraArguments >
  static std::unique_ptr< OptimizerType > createCustom( std::unique_ptr< BlackBoxProblem > problem,
                                                        const OptimizerConfiguration &configuration,
                                                        ExtraArguments &&...extra_arguments )
  {
    if ( !problem )
    {
      throw InvalidConfigurationError( "OptimizerFactory: problem cannot be null" );
    }

    const std::size_t parameter_count = problem->parametersCount();
    OptimizerConfigurationValidator::validate( configuration, parameter_count );

    std::unique_ptr< InitialParametersStrategy > initial_parameters_strategy =
      createInitialParameters( configuration.initialization, parameter_count );

    std::unique_ptr< OptimizerMethod > method = std::visit(
      [&]( const auto &hyperparameters ) -> std::unique_ptr< OptimizerMethod >
      {
        using T = std::decay_t< decltype( hyperparameters ) >;
        if constexpr ( std::is_same_v< T, HillClimbingHyperparameters > )
        {
          auto neighborhood = createNeighborhood( hyperparameters.neighborhood, configuration.random_seed );
          return std::unique_ptr< OptimizerMethod >(
            new HillClimbingMethod( hyperparameters, std::move( neighborhood ), initial_parameters_strategy.get() ) );
        }
        else if constexpr ( std::is_same_v< T, SimulatedAnnealingHyperparameters > )
        {
          auto neighborhood = createNeighborhood( hyperparameters.neighborhood, configuration.random_seed );
          auto cooling_function =
            createCoolingFunction( hyperparameters.cooling_function, hyperparameters.initial_temperature );
          return std::unique_ptr< OptimizerMethod >(
            new SimulatedAnnealingMethod( hyperparameters, std::move( neighborhood ), std::move( cooling_function ),
                                          initial_parameters_strategy.get(), configuration.random_seed ) );
        }
        else if constexpr ( std::is_same_v< T, NelderMeadHyperparameters > )
        {
          return std::unique_ptr< OptimizerMethod >(
            new NelderMeadMethod( hyperparameters, parameter_count, initial_parameters_strategy.get() ) );
        }
        else if constexpr ( std::is_same_v< T, StochasticHeuristicNelderMeadHyperparameters > )
        {
          return std::unique_ptr< OptimizerMethod >( new StochasticHeuristicNelderMeadMethod(
            hyperparameters, parameter_count, initial_parameters_strategy.get() ) );
        }
        else if constexpr ( std::is_same_v< T, ChangStochasticNelderMeadHyperparameters > )
        {
          return std::unique_ptr< OptimizerMethod >( new ChangStochasticNelderMeadMethod(
            hyperparameters, parameter_count, initial_parameters_strategy.get() ) );
        }
        else if constexpr ( std::is_same_v< T, GeneralizedPatternSearchHyperparameters > )
        {
          return std::unique_ptr< OptimizerMethod >(
            new GPSMethod( hyperparameters, initial_parameters_strategy.get() ) );
        }
        else if constexpr ( std::is_same_v< T, RinottRankingSelectionGPSHyperparameters > )
        {
          return std::unique_ptr< OptimizerMethod >(
            new RinottRankingSelectionGPSMethod( hyperparameters, initial_parameters_strategy.get() ) );
        }
        else if constexpr ( std::is_same_v< T, OpenAIESHyperparameters > )
        {
          return std::unique_ptr< OptimizerMethod >(
            new OpenAIESMethod( hyperparameters, initial_parameters_strategy.get() ) );
        }
        else if constexpr ( std::is_same_v< T, PSOHyperparameters > )
        {
          return std::unique_ptr< OptimizerMethod >(
            new PSOMethod( hyperparameters, parameter_count, initial_parameters_strategy.get() ) );
        }
        else if constexpr ( std::is_same_v< T, SPSAHyperparameters > )
        {
          return std::unique_ptr< OptimizerMethod >(
            new SPSAMethod( hyperparameters, parameter_count, initial_parameters_strategy.get() ) );
        }
        else
        {
          return std::unique_ptr< OptimizerMethod >(
            new CMAESMethod( hyperparameters, parameter_count, initial_parameters_strategy.get() ) );
        }
      },
      configuration.hyperparameters );

    return std::make_unique< OptimizerType >(
      std::move( problem ), std::move( method ), configuration, std::move( initial_parameters_strategy ),
      std::forward< ExtraArguments >( extra_arguments )... );
  }

  /**
   * @brief Creates a complete optimizer for an owned problem.
   * @param[in] problem Non-null problem whose ownership transfers to the result.
   * @param[in] configuration Complete validated settings copied into the optimizer.
   * @return Unique ownership of the assembled Optimizer.
   * @throws InvalidConfigurationError If @p problem is null or the configuration
   * violates a constraint for its problem dimension.
   */
  static std::unique_ptr< Optimizer > create( std::unique_ptr< BlackBoxProblem > problem,
                                              const OptimizerConfiguration &configuration )
  {
    return createCustom< Optimizer >( std::move( problem ), configuration );
  }
};

/** @} */

#endif // !OPTIMIZER_FACTORY_H
